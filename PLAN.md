# Batched Probing Questions

## Summary

Replace the current single-question adaptive probe loop with a **batched** probe loop. Each round, the graph generates a batch of questions (size = `min(PROBE_BATCH_SIZE, unbracketed_strand_count)`, default batch size 3), presents them to the learner, receives all answers at once, evaluates the whole batch, then either generates the next batch or exits with the boundary map. Adaptivity is preserved *between* batches via the updated boundary map; within a batch, questions are independent.

## Changes

### 1. New structured output schemas (`graphs/probe.py`)

Add two new Pydantic models alongside the existing `QuestionOut`:

```python
class BatchQuestionsOut(BaseModel):
    """LLM output for generating a batch of probe questions in one call."""
    questions: list[QuestionOut] = Field(min_length=1)

class SingleEvaluation(BaseModel):
    """Per-question evaluation result."""
    question_id: str
    is_correct: bool

class BatchEvaluateOut(BaseModel):
    """LLM output for evaluating a full batch of answers in one call."""
    evaluations: list[SingleEvaluation]
    updated_boundary_map: dict[str, dict]
    gap_summary: str
```

Remove `EvaluateOut` (replaced by `BatchEvaluateOut`). Keep `StrandsOut` and `QuestionOut` unchanged.

### 2. Graph restructure (`graphs/probe.py`)

Replace the three nodes `generate_question`, `wait_for_answer`, `evaluate_answer` with their batch counterparts. `decompose_strands` and `decide_next` stay structurally the same.

**State (`ProbeState`) changes:**
- `next_question: dict | None` → `next_batch: list[dict] | None`
- `_resume: dict | None` — now carries `{"answers": [{"question_id": str, "selected_index": int}, ...]}` instead of a single `{question_id, selected_index}`
- All other fields unchanged (`goal`, `language`, `strands`, `history`, `boundary_map`, `question_count`, `strand_descriptions`, `_decision`)

**`generate_batch` node (replaces `generate_question`):**
- Compute `batch_size = min(PROBE_BATCH_SIZE, len(unbracketed_strands))` where `unbracketed_strands` are strands with a null floor or ceiling in the current `boundary_map`.
- Single LLM call using `BatchQuestionsOut` structured output.
- Prompt: same rules as current `generate_question` (one concept per question, no confounding, even distractors, binary-search difficulty guidance), extended to "generate exactly `{batch_size}` questions. Distribute them across the unbracketed strands listed below. Multiple questions may target the same strand with escalating difficulty."
- After LLM call: for each question in the batch, generate a `uuid4` id and append the localized "I don't know" option via the existing `with_unknown_option` helper.
- Return `{"next_batch": [q1, q2, ...], "question_count": count + batch_size}`.
- Fallback: if the LLM returns fewer questions than `batch_size`, use whatever it returned (minimum 1). If it returns 0, raise 500.

**`wait_for_answers` node (replaces `wait_for_answer`):**
- `interrupt(None)` — same mechanism, just a different node name for clarity.
- Return `{"_resume": answer}` where answer is the resume payload.

**`evaluate_batch` node (replaces `evaluate_answer`):**
- Read `state["next_batch"]` and `state["_resume"]["answers"]`.
- Build a combined history summary (all questions + selected options) for the prompt.
- Single LLM call using `BatchEvaluateOut` structured output.
- Prompt: same evaluation rules as current `evaluate_answer`, extended to "evaluate all N answers. Return one evaluation per question, the full updated boundary_map, and a gap summary."
- The "I don't know" handling rule is preserved: if the learner picked the unknown option, treat as incorrect but do NOT classify `gap_type` as `"systematic"`.
- Append one history entry per question (same format as current).
- Merge boundary_map defensively (same as current).
- Return `{"boundary_map": updated, "history": history}`.

**`decide_next` node:**
- Unchanged logic: all strands bracketed OR `question_count >= MAX_PROBE_QUESTIONS` → done.
- The `bracketed()` helper stays the same.

**Graph wiring:**
```
decompose_strands → generate_batch → wait_for_answers → evaluate_batch → decide_next
    decide_next --"continue"--> generate_batch
    decide_next --"done"-------> END
```

**Env var:**
- `PROBE_BATCH_SIZE` (int, default `3`): max questions per batch.
- `MAX_PROBE_QUESTIONS` (int, default `10`): unchanged, still caps total questions.

### 3. Router / API changes (`routers/probe.py`)

**Request schema:**
```python
class AnswerIn(BaseModel):
    question_id: uuid.UUID
    selected_index: int

class ProbeIn(BaseModel):
    answers: list[AnswerIn] | None = None
```

- `answers` is `None` for the first call (start probe).
- `answers` is a non-empty list for subsequent calls (submit batch answers).

**Response schema:**
```python
class ProbeOut(BaseModel):
    phase: Phase
    questions: list[ProbeQuestionOut] | None = None
    boundary_map: dict[str, dict] | None = None
```

- `questions` replaces the old single `question` field.
- `boundary_map` and `phase` unchanged.

**Route logic (`POST /sessions/{session_id}/probe`):**

- **First call** (`body.answers is None`):
  - Same guard: session must be in `Phase.PROBING`.
  - Invoke `probe_graph` with initial state (same as current).
  - Read `result["next_batch"]` (a list of question dicts).
  - Persist all questions via `_persist_batch(db, session_id, questions)`.
  - Return `ProbeOut(phase=PROBING, questions=[ProbeQuestionOut(**q) for q in questions])`.

- **Subsequent call** (`body.answers` is a list):
  - Validate: all `question_id`s exist in the session's probe questions, none already answered, each `selected_index` is in range for its question's option count.
  - Invoke `probe_graph` with `Command(resume={"answers": [{"question_id": str(a.question_id), "selected_index": a.selected_index} for a in body.answers]})`.
  - If result has `__interrupt__`: update all answered rows via `_update_answers(db, questions, answers)`, persist new batch, return next batch.
  - If result has no `__interrupt__` (probe complete): update answered rows, set `session.boundary_map` and `session.phase = PLANNING`, return `ProbeOut(phase=PLANNING, boundary_map=...)`.

**Helper changes:**
- `_persist_question` → `_persist_batch(db, session_id, questions: list[dict])`: inserts all rows, single commit.
- `_update_answer` → `_update_answers(db, questions: list[ProbeQuestion], answers: list[tuple[uuid.UUID, int]])`: updates all rows, single commit.

### 4. DB model changes

**None.** `ProbeQuestion` already supports multiple rows per session. No schema migration needed.

### 5. Graph registry (`graphs/__init__.py`)

No changes — `build_probe_graph` is still exported and compiled the same way. The internal structure changes but the public interface (`probe_graph` with the same `invoke`/`Command(resume=...)` contract) is preserved.

## Edge Cases & Failure Modes

| Case | Handling |
|---|---|
| LLM returns fewer questions than `batch_size` | Use whatever it returned (≥1). If 0, raise 500. |
| Learner submits partial answers (missing a question from the batch) | 422: "All questions in the current batch must be answered." |
| Learner re-submits an already-answered question | 409: "Question already answered" (same as current). |
| Session not in PROBING phase | 409 (unchanged). |
| Last batch has fewer questions than `PROBE_BATCH_SIZE` | Normal — batch size is `min(PROBE_BATCH_SIZE, unbracketed_strands)`. |
| All strands bracketed after a batch | `decide_next` returns "done" → graph exits with `boundary_map`. |
| `question_count` hits `MAX_PROBE_QUESTIONS` mid-batch | Can't happen mid-batch (count increments by full batch at generation). The cap is checked in `decide_next` after evaluation. The batch that would exceed the cap is simply not generated. |

## Test Cases

1. **First call returns a batch**: POST with no `answers` → response has `questions` list of length `min(3, strand_count)`, phase still `probing`.
2. **Second call with full answers**: POST with `answers` list matching the batch → either next batch (if strands remain) or `boundary_map` (if all bracketed).
3. **Adaptivity between batches**: After a batch, the next batch's questions target only still-unbracketed strands with adjusted difficulty (verify via prompt logging or mocked LLM).
4. **Partial answer rejection**: POST with fewer `answers` than the batch size → 422.
5. **Re-answer rejection**: POST with an already-answered `question_id` → 409.
6. **Single-strand final batch**: When only 1 strand remains unbracketed, batch size = 1.
7. **MAX_PROBE_QUESTIONS cap**: After 10 total questions, next `decide_next` returns "done" even if strands aren't fully bracketed.
8. **"I don't know" option**: Each question in the batch has the localized "I don't know" option appended; selecting it is treated as incorrect but not systematic.
9. **Phase transition**: When probe completes, session phase transitions to `planning` and `boundary_map` is persisted.

## Assumptions

- The old single-question probe API is **replaced**, not kept alongside. The endpoint path and HTTP method are unchanged; only the request/response shapes differ.
- `PROBE_BATCH_SIZE` defaults to **3** (env var `PROBE_BATCH_SIZE`).
- The LLM generates the entire batch in **one structured call** (not N separate calls), and evaluates the entire batch in **one structured call**.
- All answers in a batch are **required** — no partial submission.
- `QuestionOut` (single question schema) is reused inside `BatchQuestionsOut`; its fields are unchanged.
- The `with_unknown_option` helper is applied per question in the batch (not once for the whole batch).
