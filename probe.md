# Rewrite Probe Phase as a LangGraph Interrupt/Resume Loop

## Summary

Replace the incorrect CRUD-style probe endpoints with a single `POST /sessions/{id}/probe` endpoint driven by a **Probe LangGraph** (interrupt/resume loop), matching the design in `docs/api-and-graphs.md`. The graph generates adaptive MCQs, evaluates answers, updates boundary estimates, and loops until all strands are bracketed (or a 10-question cap is hit). Each question is persisted as a `ProbeQuestion` row.

---

## Changes

### 1. `db/models.py` — Add `explanation` column to `ProbeQuestion`

Add a nullable `explanation: Mapped[str | None]` (Text) column. The LLM generates an explanation per question and it's returned in the API response; storing it gives the client a post-answer review source and matches the `StepMaterial.questions` shape.

### 2. `graphs/probe.py` — New Probe LangGraph

**State (`ProbeState` TypedDict):**
```python
class ProbeState(TypedDict, total=False):
    goal: str                    # the narrowed_goal
    history: list[dict]          # [{question_id, text, options, correct_index, selected_index, is_correct, strand, difficulty}]
    boundary_map: dict[str, dict]  # strand -> {floor, ceiling, gap_type}
    question_count: int          # how many questions asked so far
    next_question: dict | None   # the current question (output at interrupt)
```

**Nodes:**

| Node | Type | Behaviour |
|------|------|-----------|
| `generate_question` | LLM | Given `goal` + `history` + `boundary_map`, produce the next MCQ (text, 4 options, correct_index, explanation, strand, difficulty). Choose strand/difficulty to bracket the edge — escalate on all-correct, probe around misses. Sets `next_question`, increments `question_count`. |
| `wait_for_answer` | Interrupt | Pauses the graph. Resumes with `{question_id, selected_index}`. |
| `evaluate_answer` | LLM | Given the question, the learner's `selected_index`, and history: mark correct/incorrect, update `boundary_map` for the relevant strand (floor/ceiling/gap_type), append to `history`. |
| `decide_next` | Deterministic | Check: (a) all goal-relevant strands have non-null floor AND ceiling (bracketed), or (b) `question_count >= MAX_PROBE_QUESTIONS` (10). Return `"done"` or `"continue"`. |

**Control flow:**
```
generate_question → wait_for_answer (interrupt)
    → evaluate_answer → decide_next
        → "continue" → generate_question (loop)
        → "done"     → END (output boundary_map)
```

**Structured output schemas (Pydantic):**
- `QuestionOut`: `text`, `options` (list[str], length 4), `correct_index` (int), `explanation` (str), `strand` (str), `difficulty` (int 1-5)
- `EvaluateOut`: `is_correct` (bool), `updated_boundary_map` (dict[str, dict] — full map with updated strand), `gap_summary` (str, one sentence)

**Constants:**
- `MAX_PROBE_QUESTIONS = 10`

**Invocation pattern (mirrors Clarify):**
```python
# First call — no prior state
result = probe_graph.invoke(
    {"goal": narrowed_goal, "history": [], "boundary_map": {}, "question_count": 0},
    graph_config(session_id, "probe"),
)
# result contains __interrupt__ with next_question

# Subsequent calls — resume with answer
result = probe_graph.invoke(
    Command(resume={"question_id": ..., "selected_index": ...}),
    graph_config(session_id, "probe"),
)
# result contains __interrupt__ (next question) or boundary_map (done)
```

### 3. `graphs/__init__.py` — Register probe graph

Import `build_probe_graph` and compile: `probe_graph = build_probe_graph(llm, checkpointer=checkpointer)`. Export `probe_graph`.

### 4. `routers/probe.py` — Replace with single combined endpoint

**Delete** the current `GET /probe` and `POST /probe/{question_id}/answer` endpoints.

**New single endpoint:**

```
POST /sessions/{session_id}/probe
```

**Request body (Pydantic `ProbeIn`):**
```python
class ProbeIn(BaseModel):
    question_id: str | None = None
    selected_index: int | None = None
```
- First call: empty body `{}` (both fields None)
- Subsequent calls: `{question_id: "q1", selected_index: 1}`

**Response (Pydantic `ProbeOut`):**
```python
class ProbeQuestionOut(BaseModel):
    id: str
    text: str
    options: list[str]
    correct_index: int
    explanation: str
    strand: str
    difficulty: int

class ProbeOut(BaseModel):
    phase: Phase
    question: ProbeQuestionOut | None = None
    boundary_map: dict[str, dict] | None = None
```

**Handler logic:**
1. Fetch session; 404 if not found; 409 if phase ≠ `probing`.
2. If `body.question_id is None` (first call):
   - Invoke probe graph with initial state.
   - Extract `next_question` from interrupt.
   - Persist `ProbeQuestion` row (unanswered).
   - Return `ProbeOut(phase=PROBING, question=...)`.
3. If `body.question_id is not None` (subsequent call):
   - Resume graph with `Command(resume={"question_id": ..., "selected_index": ...})`.
   - If result has `__interrupt__` (next question):
     - Update the previous `ProbeQuestion` row (selected_index, is_correct, answered_at).
     - Persist new `ProbeQuestion` row.
     - Return `ProbeOut(phase=PROBING, question=...)`.
   - If result has `boundary_map` (done):
     - Update the last `ProbeQuestion` row.
     - Write `boundary_map` to `session.boundary_map`; set `session.phase = PLANNING`.
     - Return `ProbeOut(phase=PLANNING, boundary_map=...)`.

**Validation:**
- 422 if `selected_index` out of range for the question's options.
- 409 if the question was already answered (guard against double-submit).

### 5. `routers/sessions.py` — No changes needed

The clarify endpoints already transition the session to `Phase.PROBING` when the goal is narrowed. No changes required here.

---

## Response examples (matching docs)

**First call:**
```json
{
  "phase": "probing",
  "question": {
    "id": "q1",
    "text": "A 2 kg object experiences a net force of 10 N. What is its acceleration?",
    "options": ["2 m/s²", "5 m/s²", "10 m/s²", "20 m/s²"],
    "correct_index": 1,
    "explanation": "a = F/m = 10/2 = 5 m/s².",
    "strand": "newton_second_law_basic",
    "difficulty": 2
  }
}
```

**Subsequent call (next question):**
```json
{
  "phase": "probing",
  "question": { "id": "q2", "text": "...", ... }
}
```

**Final call (probe complete):**
```json
{
  "phase": "planning",
  "boundary_map": {
    "force_concept": { "floor": "...", "ceiling": null, "gap_type": "none" },
    "f_ma_relation": { "floor": null, "ceiling": "...", "gap_type": "systematic" }
  }
}
```

---

## Testing / verification

- `POST /sessions/{id}/probe` with empty body on a session in `probing` phase → returns first question.
- `POST /sessions/{id}/probe` with `{question_id, selected_index}` → returns next question.
- After 10 questions or all strands bracketed → returns `phase: "planning"` + `boundary_map`, session row updated.
- 409 if session not in probing phase.
- 404 if session doesn't exist.
- 422 if `selected_index` out of range.
- `ProbeQuestion` rows are created/updated in DB for each Q&A turn.
- Ruff + Pyright pass.

---

## Assumptions

- `explanation` is added to the `ProbeQuestion` model (nullable Text).
- The `correct_index` and `explanation` are exposed in the question response (per docs design).
- `decide_next` is a deterministic check (no LLM call), based on boundary_map completeness + question cap.
- The graph's `boundary_map` starts as `{}` and is fully managed in graph state; the final map is written to the Session row on completion.
- Question IDs are sequential (`q1`, `q2`, …) generated by the graph.
