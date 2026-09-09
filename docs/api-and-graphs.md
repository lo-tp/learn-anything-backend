# API & Graph Design

## Overview

The system is a **stateful session**. Each session walks through phases:

```
clarifying → probing → planning → reviewing → generating → executing → complete
```

All state lives in an **in-memory SQLite database** (via SQLAlchemy ORM). LangGraph graphs are invoked from FastAPI handlers to drive the LLM interactions within each phase. Swapping to a persistent DB later is a one-line connection-string change.

---

## State Schema (SQLAlchemy Models)

Database: **SQLite in-memory** (`sqlite:///:memory:`). All data is lost on process restart (acceptable for MVP). To persist later, change the connection string to `sqlite:///./learn.db` or a Postgres URL.

```python
from datetime import datetime, timezone
from sqlalchemy import create_engine, String, Text, Integer, Float, Boolean, ForeignKey, JSON
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship
from sqlalchemy.pool import StaticPool

# --- Engine (in-memory, single connection) ---

engine = create_engine(
    "sqlite:///:memory:",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,  # required for in-memory SQLite with multiple connections
)

class Base(DeclarativeBase):
    pass

# --- Enum ---

from enum import Enum

class Phase(str, Enum):
    CLARIFYING = "clarifying"
    PROBING = "probing"
    PLANNING = "planning"
    REVIEWING = "reviewing"
    GENERATING = "generating"
    EXECUTING = "executing"
    COMPLETE = "complete"

# --- Models ---

class Session(Base):
    __tablename__ = "sessions"

    session_id: Mapped[str] = mapped_column(String, primary_key=True)
    phase: Mapped[str] = mapped_column(String, default=Phase.CLARIFYING.value)
    goal: Mapped[str] = mapped_column(Text)
    narrowed_goal: Mapped[str | None] = mapped_column(Text, nullable=True)
    boundary_map: Mapped[dict | None] = mapped_column(JSON, nullable=True)  # strand -> {floor, ceiling, gap_type}
    created_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc))
    updated_at: Mapped[datetime] = mapped_column(default=lambda: datetime.now(timezone.utc), onupdate=lambda: datetime.now(timezone.utc))

    # Relationships
    probe_questions: Mapped[list["ProbeQuestion"]] = relationship(back_populates="session", cascade="all, delete")
    plan: Mapped["Plan | None"] = relationship(back_populates="session", uselist=False, cascade="all, delete")
    materials: Mapped[list["StepMaterial"]] = relationship(back_populates="session", cascade="all, delete")
    step_progress: Mapped[list["StepProgress"]] = relationship(back_populates="session", cascade="all, delete")


class ProbeQuestion(Base):
    __tablename__ = "probe_questions"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    question_id: Mapped[str] = mapped_column(String)          # stable ID within the probe sequence
    text: Mapped[str] = mapped_column(Text)
    options: Mapped[list] = mapped_column(JSON)               # list[str]
    correct_index: Mapped[int] = mapped_column(Integer)
    strand: Mapped[str] = mapped_column(String)               # which prerequisite strand
    difficulty: Mapped[int] = mapped_column(Integer)          # 1-5
    # Answer (null until answered)
    selected_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    answered_at: Mapped[datetime | None] = mapped_column(nullable=True)

    session: Mapped["Session"] = relationship(back_populates="probe_questions")


class Plan(Base):
    __tablename__ = "plans"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), unique=True)
    version: Mapped[int] = mapped_column(Integer, default=1)  # increments on each adjustment
    prose_summary: Mapped[str] = mapped_column(Text)
    dependency_dag: Mapped[str] = mapped_column(Text)          # mermaid source
    steps: Mapped[list] = mapped_column(JSON)                  # list of {id, title, description, depends_on, depth, status}
    adjustments: Mapped[list] = mapped_column(JSON, default=list)  # history of adjustment texts

    session: Mapped["Session"] = relationship(back_populates="plan")


class StepMaterial(Base):
    __tablename__ = "step_materials"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    step_id: Mapped[str] = mapped_column(String)
    slides: Mapped[list] = mapped_column(JSON)                 # list of slide IDs (references to sandbox service)
    questions: Mapped[list] = mapped_column(JSON)              # list of {id, text, options, correct_index, explanation}
    summary: Mapped[dict] = mapped_column(JSON)                # ConceptSummary: {step_id, title, key_points: [str]}

    session: Mapped["Session"] = relationship(back_populates="materials")


class StepProgress(Base):
    __tablename__ = "step_progress"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    session_id: Mapped[str] = mapped_column(ForeignKey("sessions.session_id"), index=True)
    step_id: Mapped[str] = mapped_column(String)
    slides_done: Mapped[bool] = mapped_column(Boolean, default=False)
    answers: Mapped[list] = mapped_column(JSON, default=list)  # list of {question_id, selected_index, is_correct, explanation}
    score: Mapped[float | None] = mapped_column(Float, nullable=True)
    complete: Mapped[bool] = mapped_column(Boolean, default=False)

    session: Mapped["Session"] = relationship(back_populates="step_progress")


# --- Init ---

Base.metadata.create_all(engine)
```

---

## Endpoints

### Session lifecycle

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions` | Create session, start goal capture |
| `GET` | `/sessions/{id}` | Get current session state & progress |
| `POST` | `/sessions/{id}/clarify` | Submit clarification answer (if goal was too broad) |

### Probe phase

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions/{id}/probe` | Start probe / submit next answer (combined) |

### Plan phase

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions/{id}/plan/generate` | Trigger plan generation (auto after probe or explicit) |
| `POST` | `/sessions/{id}/plan/adjust` | Submit a free-text adjustment, get regenerated plan |
| `POST` | `/sessions/{id}/plan/approve` | Approve plan, trigger material generation |

### Execution phase (no LLM — deterministic only)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/sessions/{id}/steps/{step_id}` | Fetch step manifest: slide IDs + questions |
| `POST` | `/sessions/{id}/steps/{step_id}/answers` | Submit all answers → get results, score, pass/fail, next step |

> **Note:** Slide HTML content is served by a separate sandbox service (out of scope). The `slides` array in the step manifest contains IDs that the client uses to fetch individual slides from the sandbox.

### Endpoint details

#### `POST /sessions`

```json
// Request
{ "goal": "I want to learn Newton's second law of motion." }

// Response (goal is specific enough)
{
  "session_id": "abc123",
  "phase": "probing",
  "narrowed_goal": "Newton's second law of motion (F = ma)"
}

// Response (goal too broad)
{
  "session_id": "abc123",
  "phase": "clarifying",
  "clarifying_questions": [
    "Do you want to focus on the mathematical formulation, the physical intuition, or applications?",
    "Do you already know about force, mass, and acceleration as separate concepts?"
  ]
}
```

#### `POST /sessions/{id}/clarify`

```json
// Request
{ "answer": "I want the physical intuition and some math. I know what force and mass are but not acceleration formally." }

// Response
{
  "session_id": "abc123",
  "phase": "probing",
  "narrowed_goal": "Newton's second law: physical intuition + mathematical formulation. Learner knows force and mass, needs formal acceleration."
}
```

#### `POST /sessions/{id}/probe`

This is the **combined** probe endpoint. First call has no answer (starts the probe); subsequent calls submit the answer to the last question.

```json
// Request (first call — start probe)
{ }

// Response
{
  "phase": "probing",
  "question": {
    "id": "q1",
    "text": "A 2 kg object experiences a net force of 10 N. What is its acceleration?",
    "options": ["2 m/s²", "5 m/s²", "10 m/s²", "20 m/s²"],
    "correct_index": 1,
    "explanation": "a = F/m = 10/2 = 5 m/s². Acceleration is force divided by mass.",
    "strand": "newton_second_law_basic",
    "difficulty": 2
  },
  "questions_remaining": 9
}

// Request (subsequent call — submit answer)
{ "question_id": "q1", "selected_index": 1 }

// Response (next question)
{
  "phase": "probing",
  "previous": {
    "question_id": "q1",
    "was_correct": true
  },
  "question": {
    "id": "q2",
    "text": "If the mass of an object doubles while the net force stays the same, what happens to its acceleration?",
    "options": ["It doubles", "It halves", "It stays the same", "It quadruples"],
    "correct_index": 1,
    "explanation": "F = ma, so a = F/m. If m doubles and F stays the same, a halves.",
    "strand": "newton_second_law_proportionality",
    "difficulty": 3
  },
  "questions_remaining": 8
}

// Response (probe complete — edge bracketed on all strands)
{
  "phase": "planning",
  "boundary_map": {
    "force_concept": { "floor": "Understands force as a push/pull with magnitude and direction", "ceiling": null, "gap_type": "none" },
    "mass_concept": { "floor": "Knows mass as measure of matter", "ceiling": null, "gap_type": "none" },
    "acceleration_concept": { "floor": "Knows acceleration informally as 'speeding up'", "ceiling": "Doesn't know it's a vector / rate of change of velocity", "gap_type": "narrow" },
    "f_ma_relation": { "floor": null, "ceiling": "Hasn't seen F=ma as a unified equation", "gap_type": "systematic" }
  },
  "message": "Probe complete. Generating plan…"
}
```

#### `POST /sessions/{id}/plan/generate`

```json
// Request
{ }

// Response
{
  "phase": "reviewing",
  "plan": {
    "prose_summary": "We'll start by formalising acceleration as a vector (your gap), then revisit force and mass precisely, then combine them into F=ma, then explore applications.",
    "dependency_dag": "graph LR\n  A[Formal acceleration] --> C[F=ma]\n  B[Force & mass review] --> C\n  C --> D[Applications & worked examples]",
    "steps": [
      { "id": "s1", "title": "Acceleration as a vector", "description": "Define a = dv/dt, vector nature, units", "depends_on": [], "depth": 2 },
      { "id": "s2", "title": "Force and mass — precise definitions", "description": "Net force, inertia, kg as unit", "depends_on": [], "depth": 1 },
      { "id": "s3", "title": "Newton's second law: F = ma", "description": "Derive the relation, vector equation, units check", "depends_on": ["s1", "s2"], "depth": 3 },
      { "id": "s4", "title": "Applications and worked examples", "description": "Inclined planes, friction, connected bodies", "depends_on": ["s3"], "depth": 3 }
    ]
  }
}
```

#### `POST /sessions/{id}/plan/adjust`

```json
// Request
{ "adjustment": "I don't know what gravity is — add a step before force definitions that covers gravitational force." }

// Response (full plan regenerated)
{
  "phase": "reviewing",
  "plan": {
    "prose_summary": "…",
    "dependency_dag": "graph LR\n  G[Gravity & gravitational force] --> B[Force & mass review]\n  A[Formal acceleration] --> C[F=ma]\n  B --> C\n  C --> D[Applications]",
    "steps": [
      { "id": "s0", "title": "Gravitational force", "description": "What gravity is, g ≈ 9.8 m/s², weight vs mass", "depends_on": [], "depth": 2 },
      { "id": "s1", "title": "Acceleration as a vector", "depends_on": [], "depth": 2 },
      { "id": "s2", "title": "Force and mass — precise definitions", "depends_on": ["s0"], "depth": 1 },
      { "id": "s3", "title": "Newton's second law: F = ma", "depends_on": ["s1", "s2"], "depth": 3 },
      { "id": "s4", "title": "Applications and worked examples", "depends_on": ["s3"], "depth": 3 }
    ]
  }
}
```

#### `POST /sessions/{id}/plan/approve`

```json
// Request
{ }

// Response (materials generated synchronously)
{
  "phase": "executing",
  "current_step_id": "s0",
  "total_steps": 5,
  "message": "Plan approved. Materials generated. Start with step 's0'."
}
```

#### `GET /sessions/{id}/steps/{step_id}`

Returns the step manifest: slide IDs (for the sandbox service) and questions.

```json
// Response
{
  "step_id": "s0",
  "title": "Gravitational force",
  "slides": ["s0_slide_1", "s0_slide_2", "s0_slide_3"],
  "questions": [
    {
      "id": "q1",
      "text": "An object has a mass of 5 kg. What is its weight on Earth (g = 9.8 m/s²)?",
      "options": ["5 N", "49 N", "9.8 N", "50 N"],
      "correct_index": 1,
      "explanation": "Weight = mg = 5 × 9.8 = 49 N. Mass is in kg; weight is a force in newtons."
    },
    {
      "id": "q2",
      "text": "True or False: An object's weight is the same on the Moon as on Earth.",
      "options": ["True", "False"],
      "correct_index": 1,
      "explanation": "Weight = mg, and g is smaller on the Moon (~1.6 m/s²). Mass stays the same; weight changes."
    }
  ]
}
```

The client uses the `slides` IDs to fetch individual slide HTML from the sandbox service (separate system, out of scope).

#### `POST /sessions/{id}/steps/{step_id}/answers`

Submit all answers for the step in one batch. No LLM — deterministic index comparison.

```json
// Request
{
  "answers": [
    { "question_id": "q1", "selected_index": 1 },
    { "question_id": "q2", "selected_index": 0 }
  ]
}

// Response (passed — more steps remain)
{
  "step_id": "s0",
  "results": [
    { "question_id": "q1", "correct": true,  "explanation": "Weight = mg = 5 × 9.8 = 49 N." },
    { "question_id": "q2", "correct": false, "explanation": "Weight = mg, and g is smaller on the Moon (~1.6 m/s²). Mass stays the same; weight changes." }
  ],
  "score": 0.5,
  "passed": false,
  "revisit": "Review the slides on weight vs mass before retrying.",
  "progress": { "completed": 0, "total": 5 }
}

// Response (passed — advance to next step)
{
  "step_id": "s0",
  "results": [...],
  "score": 1.0,
  "passed": true,
  "next_step_id": "s1",
  "progress": { "completed": 1, "total": 5 }
}

// Response (passed — all done)
{
  "step_id": "s4",
  "results": [...],
  "score": 1.0,
  "passed": true,
  "phase": "complete",
  "progress": { "completed": 5, "total": 5 },
  "message": "🎉 You've completed the full plan. Mastery achieved."
}
```

#### `GET /sessions/{id}`

```json
// Response
{
  "session_id": "abc123",
  "phase": "executing",
  "narrowed_goal": "Newton's second law…",
  "progress": {
    "current_step_id": "s2",
    "completed_steps": ["s0", "s1"],
    "total_steps": 5,
    "step_scores": { "s0": 1.0, "s1": 0.75 }
  }
}
```

---

## LangGraph Graphs

Three graphs. Each is invoked from a FastAPI handler. State is persisted in the in-memory DB between invocations (the graph itself is stateless per-invocation; we pass/retrieve state via the session store). Execution (step 6) needs no graph — it's a DB read + index comparison.

### Graph 1: Probe Graph

**Purpose:** Adaptive questioning loop that brackets the learner's boundary.

**Entry:** `(narrowed_goal, prior_answers)`  
**Exit:** `boundary_map` (when done) or `next_question` (when continuing)

```
┌─────────────────────────────────────────────────────────┐
│                    PROBE GRAPH                           │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │ generate_     │     │ evaluate_    │                 │
│  │ question      │◄───│ answer       │                 │
│  │              │     │              │                 │
│  └──────┬───────┘     └──────┬───────┘                 │
│         │                     │                         │
│         │ (return q to        │                         │
│         │  caller/interrupt)  │                         │
│         ▼                     ▼                         │
│    [interrupt]          ┌──────────────┐               │
│                         │ decide_next  │               │
│                         │              │               │
│                         └──┬───────┬───┘               │
│                            │       │                    │
│                   (ask more)│       │(edge bracketed)   │
│                            │       ▼                    │
│                            │  ┌──────────┐             │
│                            └──│  END     │             │
│                               │(boundary)│             │
│                               └──────────┘             │
└─────────────────────────────────────────────────────────┘
```

**Nodes:**

| Node | LLM task |
|------|----------|
| `generate_question` | Given goal + answer history + current boundary estimates, produce the next MCQ/TF question. Choose strand and difficulty to bracket the edge. Escalate if all-correct; probe around if a miss. |
| `evaluate_answer` | Given the question, the learner's answer, and history: mark correct/incorrect, update boundary estimates for the relevant strand, characterise gap type. |
| `decide_next` | Given updated boundary: are all goal-relevant strands bracketed? Have we hit 10 questions? → return `"continue"` or `"done"`. |

**Control flow:**
- `generate_question` → **interrupt** (return question to FastAPI handler, which returns it to the client)
- On next call with answer: `evaluate_answer` → `decide_next` → conditional edge:
  - `"continue"` → `generate_question` → interrupt
  - `"done"` → END (output `boundary_map`)

**Invocation pattern from FastAPI:**
```python
# First call (no answer):
result = probe_graph.invoke({
    "goal": session["narrowed_goal"],
    "history": [],
    "answer": None
})
# result = {"next_question": {...}}

# Subsequent calls:
result = probe_graph.invoke({
    "goal": session["narrowed_goal"],
    "history": session["probe_answers"],
    "answer": {"question_id": ..., "selected_index": ...}
})
# result = {"next_question": {...}} or {"boundary_map": {...}}
```

---

### Graph 2: Plan Graph

**Purpose:** Generate the learning plan, then loop on user adjustments until approved.

**Entry:** `(narrowed_goal, boundary_map)`  
**Exit:** Approved `Plan` (steps + prose + DAG)

The graph loops: generate/refine → interrupt (show plan to user) → user approves or adjusts → if adjust, loop back with the adjustment. On the first pass, the full research → design → render pipeline runs. On subsequent passes (refinements), `research_topic` is skipped — the topic hasn't changed, only the plan structure.

```
┌──────────────────────────────────────────────────────────────────┐
│                       PLAN GRAPH                                  │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌───────────┐          │
│  │ research_    │───►│ design_      │───►│ render_   │          │
│  │ topic        │    │ plan         │    │ plan      │          │
│  └──────────────┘    └──────────────┘    └───────────┘          │
│        (first         (build/refine    (format as steps,        │
│         pass only)     step list with   prose summary,          │
│                        deps, order)     mermaid DAG)            │
│                                                                  │
│                                      ┌──────────┐               │
│                                      │[interrupt]│               │
│                                      └────┬─────┘               │
│                                           │                     │
│                                  (user response)                 │
│                                           │                     │
│                              ┌────────────┴────────────┐        │
│                              │                         │        │
│                         "approve"                 "adjust"     │
│                              │                         │        │
│                              ▼                         │        │
│                         ┌──────────┐                  │        │
│                         │   END    │                  │        │
│                         │(approved)│                  │        │
│                         └──────────┘                  │        │
│                                                      │        │
│                                                      │        │
│                    (loop back with adjustment as     │        │
│                     context; skip research)          │        │
│                                                      │        │
│  ┌──────────────────────────────────────────────────┘        │
│  │  (re-enter at design_plan)                                 │
└────────────────────────────────────────────────────────────────┘
```

**Nodes:**

| Node | LLM task | Runs |
|------|----------|------|
| `research_topic` | Map the topic: core concepts, first principles, standard framings, common gotchas. Identify the unconditional truths. (This is the "scope the field" step from teacher.md.) | First pass only |
| `design_plan` | Given research + boundary_map + current_plan (baseline, if refining) + adjustment (if refining): determine which truths the learner already has, build the dependency-ordered step list from their boundary to the goal. If a `current_plan` exists, use it as the starting structure and apply the adjustment (add/remove/reorder steps, adjust depth). Each step is small, focused, and explicitly dependent on prior steps. | Every pass |
| `render_plan` | Format the plan: write the prose summary (why this order, given their starting point), generate the Mermaid DAG, assign step IDs and metadata. | Every pass |

**Control flow:** Loop with interrupt. After `render_plan`, the graph **interrupts** and returns the plan to the client. On the next call, the user's response determines the route:
- `"approve"` → END (plan is final, trigger material generation)
- `"adjust: <text>"` → loop back to `design_plan` (skip `research_topic`)

**Invocation pattern from FastAPI:**
```python
# Initial generation (first call, no prior plan):
result = plan_graph.invoke({
    "goal": session["narrowed_goal"],
    "boundary_map": session["boundary_map"],
    "response": None  # first call, no user response yet
})
# result = { "plan": {...} }  (graph is now paused at interrupt)

# User adjusts (subsequent call):
result = plan_graph.invoke({
    "goal": session["narrowed_goal"],
    "boundary_map": session["boundary_map"],
    "response": { "action": "adjust", "text": "I don't know what gravity is — add a step…" }
})
# result = { "plan": {...} }  (graph interrupts again with refined plan)

# User approves (final call):
result = plan_graph.invoke({
    "goal": session["narrowed_goal"],
    "boundary_map": session["boundary_map"],
    "response": { "action": "approve" }
})
# result = { "plan": {...}, "approved": True }  (graph reaches END)
```

**State persisted across the loop (via LangGraph checkpointer, keyed by `session_id`):**
- `current_plan` — the last rendered plan (baseline for next refinement)
- `research` — the topic research output (reused across refinements)
- `pass_count` — how many refinement passes have occurred

---

### Graph 3: Material Generation Graph

**Purpose:** For a single approved plan step, generate the HTML slides, quiz questions, and a compact summary of established concepts (for use by subsequent steps).

**Entry:** `(step, established_concepts, learner_context)`  
**Exit:** `StepMaterial` (slides + questions) + `ConceptSummary` (key points for downstream steps)

**Context strategy:** To avoid unbounded context growth, each step receives only a **cumulative concept summary** (a few key points per prior step), not the full HTML slides. This keeps input size O(n × small_constant) regardless of plan length.

```
┌──────────────────────────────────────────────────────────────────┐
│              MATERIAL GENERATION GRAPH                           │
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐       │
│  │ write_slides │───►│ write_       │───►│ summarize_   │       │
│  │              │    │ questions    │    │ step         │       │
│  └──────────────┘    └──────────────┘    └──────────────┘       │
│                                                                  │
│  (HTML slides        (MCQ/TF questions   (3-5 key points:       │
│   following           with explanations,  definitions,          │
│   teacher.md          following         formulas, key           │
│   principles)         teacher.md        insights — used         │
│                       option-           as context for          │
│                       construction      subsequent steps)       │
│                       procedure)                                  │
└──────────────────────────────────────────────────────────────────┘
```

**Nodes:**

| Node | LLM task |
|------|----------|
| `write_slides` | Write HTML slides for this step. Follow teacher.md: motivate the concept, establish unconditional truths before building, show the dependency connection to prior steps (referenced via `established_concepts`). Use LaTeX for math. Multiple slides (3-7 typical). Generated HTML is pushed to the sandbox service; this node returns the assigned slide IDs. |
| `write_questions` | Write 3-5 MCQ/TF questions testing that the step's concepts landed. Follow the option-construction procedure in teacher.md (bare claims, mutate correct → distractors, no asymmetric bolding, explanations separate). |
| `summarize_step` | Given the generated slides, extract 3-5 key points: definitions, formulas, core insights. Output is a compact `ConceptSummary` object stored for use by subsequent steps. |

**Control flow:** Linear. Called once per step, sequentially through the plan (step 1 → step 2 → … → step N). Each call receives the **accumulated concept summaries** from all prior steps (not the full materials), so context stays small.

**ConceptSummary shape:**
```python
class ConceptSummary(TypedDict):
    step_id: str
    title: str
    key_points: list[str]   # 3-5 items: "F_g = mg", "weight ≠ mass", "a = dv/dt", etc.
```

**Invocation pattern from FastAPI:**
```python
# Called during plan/approve, sequentially:
summaries: list[ConceptSummary] = []

for step in plan["steps"]:
    result = material_graph.invoke({
        "step": step,
        "established_concepts": summaries,  # compact, stays small
        "learner_context": session["boundary_map"]
    })
    # result = {
    #   "slides": ["s0_slide_1", "s0_slide_2", "s0_slide_3"],  # IDs; HTML pushed to sandbox
    #   "questions": [...],
    #   "summary": {...}
    # }
    # (The write_slides node pushes generated HTML to the sandbox service
    #  and returns the assigned slide IDs.)
    db.add(StepMaterial(
        session_id=session_id,
        step_id=step["id"],
        slides=result["slides"],
        questions=result["questions"],
        summary=result["summary"]
    ))
    summaries.append(result["summary"])
```

**Why not pass full prior materials?**

| Approach | Context at step 8 (est.) | Scales? |
|----------|--------------------------|--------|
| Full prior materials (HTML + questions) | ~30-40k tokens | No — grows linearly with plan length |
| Cumulative concept summaries | ~1-2k tokens | Yes — a few lines per step, bounded |

---

---

## Graph Invocation Flow (end-to-end)

```
Client                    FastAPI Handler              LangGraph
  │                           │                           │
  │── POST /sessions ────────►│                           │
  │                           │── (simple LLM call) ────►│  [clarify: is goal specific?]
  │◄── session_id, phase ────│                           │
  │                           │                           │
  │── POST /probe ───────────►│── invoke ───────────────►│  [Probe Graph: generate_question]
  │◄── question 1 ───────────│                           │
  │                           │                           │
  │── POST /probe {ans} ────►│── invoke ───────────────►│  [Probe Graph: evaluate → decide]
  │◄── question 2 ───────────│                           │
  │         ...              │         ...               │
  │◄── boundary_map ─────────│◄── END ──────────────────│  [Probe Graph: done]
  │                           │                           │
  │── POST /plan/generate ──►│── invoke ───────────────►│  [Plan Graph: research → design → render]
  │◄── plan ─────────────────│                           │
  │                           │                           │
  │── POST /plan/adjust ────►│── invoke ───────────────►│  [Plan Graph: (re)generate with adjustment]
  │◄── regenerated plan ─────│                           │
  │         ...              │         ...               │
  │                           │                           │
  │── POST /plan/approve ───►│── invoke (per step) ────►│  [Material Graph × N steps]
  │◄── "ready, start s0" ───│                           │
  │                           │                           │
  │── GET /steps/s0 ────────►│  (DB lookup)              │
  │◄── slide IDs + questions─│                           │
  │                           │                           │
  │── (client fetches slides from sandbox service)       │
  │         ...              │         ...               │
  │                           │                           │
  │── POST /steps/s0/answers►│  (index comparison,       │
  │◄── results + score ──────│   no LLM)                 │
  │         ...              │         ...               │
  │◄── "🎉 mastery achieved" │                           │
```

---

## Summary Table

| # | Graph | Phases served | LLM calls per invocation | Loop? |
|---|-------|---------------|--------------------------|-------|
| 1 | **Probe** | Probing | 1 (generate) or 2 (evaluate + decide) | Yes — interrupt/resume loop |
| 2 | **Plan** | Planning, Reviewing | 2-3 (research once + design + render per pass) | Yes — interrupt/resume loop until approved |
| 3 | **Material** | Generating (on approve) | 3 (slides + questions + summary) per step | No — linear, called N times |

Execution (step 6) requires **no graph and no LLM** — it's a DB read + index comparison.

---

## Implementation Notes

1. **Probe graph uses LangGraph interrupts.** The `generate_question` node emits the question and the graph pauses. The next `POST /probe` call resumes with the answer. Use `graph.invoke()` with a thread/checkpointer keyed by `session_id` to persist the pause state in memory.

2. **Plan graph uses LangGraph interrupts (same pattern as Probe).** The graph pauses after `render_plan` and resumes when the user's next response arrives (adjust or approve). The `current_plan` and `research` are persisted in the graph state across the loop. On approval, the final plan is written to the `Plan` row (same `session_id`, incremented `version` on each pass).

3. **Material generation is sequential and must stay sequential.** Each step's `write_slides` node needs the accumulated `ConceptSummary` list from all prior steps, so steps must be generated in order. The context stays small (~1-2k tokens for the summaries) regardless of plan length. For large plans, consider a background task + polling endpoint so the client isn't waiting synchronously.

4. **Execution is purely deterministic.** No LLM, no graph. The `POST .../answers` handler compares `selected_index` to `correct_index` for each question, computes the score, and returns results. The "revisit" message on failure is a static template (e.g. "Review the slides before retrying"), not an LLM-generated suggestion.

5. **All state is in an in-memory SQLite DB.** No persistent storage. On process restart, all sessions are lost (acceptable for MVP). To persist, change the connection string: `sqlite:////absolute/path/to/learn.db` or a Postgres URL. The ORM models don't change.

6. **Use a session-scoped DB session per request.** FastAPI dependency:
```python
from sqlalchemy.orm import Session as DBSession
from sqlalchemy.orm import sessionmaker

SessionFactory = sessionmaker(bind=engine)

def get_db() -> DBSession:
    db = SessionFactory()
    try:
        yield db
    finally:
        db.close()
```
Each handler receives `db: DBSession` as a parameter. Commit explicitly after mutations.
