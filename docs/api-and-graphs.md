# API & Graph Design

## Overview

The system is a **stateful session**. Each session walks through phases:

```
clarifying → probing → planning → reviewing → generating → executing → complete
```

State lives in **two stores** with distinct responsibilities:

1. **Domain state** — the `Session` row, its phase, probe questions, plan, materials, and step progress — lives in an **in-memory SQLite database** (via SQLAlchemy ORM). This is the durable record the API reads and writes.
2. **Graph execution state** — each LangGraph graph's in-flight state, including exactly where it is paused at an interrupt and intermediate node outputs (e.g. the plan's `current_plan` baseline, the `research` output, mid-loop boundary estimates) — lives in a **LangGraph checkpointer**, keyed by `thread_id = f"{session_id}:{graph_name}"`. The checkpointer is what lets the Clarify, Probe, and Plan graphs *pause at an interrupt and resume on the next request* without the handler re-deriving state. For now it is **in-memory** (`MemorySaver`), matching the in-memory domain DB.

LangGraph graphs are invoked from FastAPI handlers to drive the LLM interactions within each phase.  **Both stores are in-memory for the MVP and will later migrate to Postgres together** — the domain DB via a connection-string change, and the checkpointer via `PostgresSaver` (a drop-in replacement for `MemorySaver`).

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

## Graph Checkpointer

The **domain DB** (above) holds the session's durable record. LangGraph needs a second store — a **checkpointer** — to hold each graph's *execution* state across calls. This is what makes interrupt/resume work. For now we use the in-memory `MemorySaver`, consistent with the in-memory domain DB; both migrate to Postgres later (see note 6).

**Thread identity:** `thread_id = f"{session_id}:{graph_name}"`. A single session runs **three** different graphs (Clarify, Probe, Plan), and the checkpointer namespaces *all* checkpoints by `thread_id`. Reusing a bare `session_id` for all of them would make their pause states **collide and overwrite each other**. Each graph therefore gets its own thread — `abc123:clarify`, `abc123:probe`, `abc123:plan` — so each loop pauses and resumes independently within the same session.

```python
from langgraph.checkpoint.memory import MemorySaver

# One in-memory checkpoint store shared by every graph (MVP).
# Lives for the process lifetime; lost on restart — same as the domain DB.
checkpointer = MemorySaver()

def graph_config(session_id: str, graph: str) -> dict:
    # One distinct checkpoint thread per (session, graph). A session runs the
    # Clarify, Probe, and Plan graphs, so they must NOT share a thread — each
    # gets its own namespace, e.g. "abc123:clarify", "abc123:probe", "abc123:plan".
    return {"configurable": {"thread_id": f"{session_id}:{graph}"}}

# Run a graph for a session (the checkpointer is bound at compile time):
def run_graph(graph, state, session_id: str, graph_name: str):
    return graph.invoke(state, graph_config(session_id, graph_name))
```

Graphs are compiled with the checkpointer:

```python
clarify_graph  = clarify_graph_fn(checkpointer=checkpointer)
probe_graph    = probe_graph_fn(checkpointer=checkpointer)
plan_graph     = plan_graph_fn(checkpointer=checkpointer)
material_graph = material_graph_fn(checkpointer=checkpointer)
```

**What the checkpointer stores (per thread):**
- The graph state dict at each super-step (node outputs accumulated so far)
- The pending node(s) — i.e. exactly where the graph is paused at an interrupt
- The full checkpoint history (enables replay / time-travel debugging)

**What it does NOT replace — the domain DB.** The checkpointer holds *graph-internal* state (e.g. `current_plan`, `research`, mid-loop boundary estimates). When a graph produces a durable artifact (a `Plan` row, a `StepMaterial` row, the final `boundary_map`), the handler writes it to the SQLAlchemy DB so it is queryable by the API and independent of the graph.

> **Migration path (later):** when the domain DB moves to Postgres, swap `MemorySaver` for `PostgresSaver` (`pip install langgraph-checkpoint-postgres`) — it implements the same checkpointer interface, so only the construction changes: `checkpointer = PostgresSaver(conn)`. The per-graph `thread_id = f"{session_id}:{graph_name}"` keying and all invocation code stay identical.

---

## Endpoints

### Session lifecycle

| Method | Path | Purpose |
|--------|------|---------|
| `POST` | `/sessions` | Create session, start goal capture |
| `GET` | `/sessions` | List all sessions (newest first), optionally filtered by one or more `?phase=` values |
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
| `POST` | `/sessions/{id}/plan/approve` | Approve plan, kick off background material generation (returns `202` immediately) |
| `GET` | `/sessions/{id}/materials` | Poll material generation progress (step IDs generated so far) |

### Execution phase (no LLM — deterministic only)

| Method | Path | Purpose |
|--------|------|---------|
| `GET` | `/sessions/{id}/steps/{step_id}` | Fetch step manifest: slide IDs + questions |
| `POST` | `/sessions/{id}/steps/{step_id}/answers` | Submit all answers → get results, score, pass/fail, next step |

> **Note:** Slide HTML content is served by a separate sandbox service (out of scope). The `slides` array in the step manifest contains IDs that the client uses to fetch individual slides from the sandbox.

### Endpoint details

#### `GET /sessions`

Lists all sessions, newest `created_at` first. Filter with a repeatable `phase` query parameter (OR semantics), e.g. `GET /sessions?phase=generating&phase=probing`. An invalid phase value returns `422` (enum validation).

```json
// Response
{
  "sessions": [
    {
      "session_id": "abc123",
      "phase": "probing",
      "goal": "I want to learn Newton's second law of motion.",
      "narrowed_goal": "Newton's second law: physical intuition + mathematical formulation.",
      "created_at": "2026-09-11T07:00:00Z"
    },
    {
      "session_id": "def456",
      "phase": "executing",
      "goal": "I want to learn integration by parts.",
      "narrowed_goal": null,
      "created_at": "2026-09-10T15:30:00Z"
    }
  ]
}
```

#### `POST /sessions`

Creates the session and makes the **first call to the Clarify Graph** (`thread_id = f"{session_id}:clarify"`). If the raw goal is already specific enough, the graph ends immediately and the session advances to `probing`; otherwise it interrupts with clarifying questions and the session sits in `clarifying`.

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

Drives the **Clarify Graph** loop (`thread_id = f"{session_id}:clarify"`). Each call resumes the graph with the user's answer; the loop continues until the goal is specific enough. So the response is either the narrowed goal (hand-off to Probe) **or** another round of clarifying questions.

```json
// Request
{ "answer": "I want the physical intuition and some math. I know what force and mass are but not acceleration formally." }

// Response A (goal now specific enough — hand off to Probe)
{
  "session_id": "abc123",
  "phase": "probing",
  "narrowed_goal": "Newton's second law: physical intuition + mathematical formulation. Learner knows force and mass, needs formal acceleration."
}

// Response B (still too broad — another clarification round)
{
  "session_id": "abc123",
  "phase": "clarifying",
  "clarifying_questions": [
    "Do you want worked-example practice, or just the derivation and intuition?"
  ]
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
  }
}

// Request (subsequent call — submit answer)
{ "question_id": "q1", "selected_index": 1 }

// Response (next question)
{
  "phase": "probing",
  "question": {
    "id": "q2",
    "text": "If the mass of an object doubles while the net force stays the same, what happens to its acceleration?",
    "options": ["It doubles", "It halves", "It stays the same", "It quadruples"],
    "correct_index": 1,
    "explanation": "F = ma, so a = F/m. If m doubles and F stays the same, a halves.",
    "strand": "newton_second_law_proportionality",
    "difficulty": 3
  }
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

Four graphs. Each is invoked from a FastAPI handler. State is persisted in the in-memory DB between invocations (the graph itself is stateless per-invocation; we pass/retrieve state via the session store). Three of the four — **Clarify**, **Probe**, and **Plan** — are interrupt/resume **loops** held by the in-memory checkpointer; **Material** is linear. Execution (step 6) needs no graph — it's a DB read + index comparison.

### Graph 1: Clarify Graph

**Purpose:** Iteratively narrow a possibly-broad goal into a precise, learnable `narrowed_goal`. Loops on user clarification until the goal is specific enough (or a safety cap is hit), then hands off to Probe.

**Entry:** `(goal, prior_clarifications)`  
**Exit:** `narrowed_goal` (when specific enough) or `clarifying_questions` (when another round is needed)

```
┌─────────────────────────────────────────────────────────┐
│                   CLARIFY GRAPH                         │
│                                                         │
│  ┌──────────────┐     ┌──────────────┐                 │
│  │ assess_      │     │ refine_goal  │                 │
│  │ goal         │     │              │                 │
│  │              │     │              │                 │
│  └──┬───────┬───┘     └──────┬───────┘                 │
│     │       │                │                         │
│  (specific)│  (too broad)   │                          │
│     │       ▼                │                          │
│     │  ┌──────────────┐      │                          │
│     │  │ generate_    │      │                          │
│     │  │ questions    │      │                          │
│     │  └──────┬───────┘      │                          │
│     │         │              │                          │
│     │         ▼              │                          │
│     │    [interrupt]         │                          │
│     │    (return qs)         │                          │
│     │         │              │                          │
│     │   (user answer)        │                          │
│     │         │              │                          │
│     │         └──────────────┘  (loop back to assess_   │
│     │                      goal with new context)       │
│     ▼                                                   │
│  ┌──────────┐                                          │
│  │   END    │  (output narrowed_goal)                  │
│  │(narrowed)│                                          │
│  └──────────┘                                          │
└─────────────────────────────────────────────────────────┘
```

**Nodes:**

| Node | LLM task |
|------|----------|
| `assess_goal` | Given the goal + all prior clarification answers, judge whether it is specific enough to plan against. Output: `"specific"` (with the `narrowed_goal`) or `"too_broad"` (with the dimensions still open). |
| `generate_questions` | Given the open dimensions, produce 1-3 targeted clarifying questions that will most reduce the remaining ambiguity. |
| `refine_goal` | Given the goal + the latest answer, fold it into a tighter working goal and update the set of open dimensions (feeds the next `assess_goal`). |

**Control flow:**
- `assess_goal` → conditional:
  - `"specific"` → END (output `narrowed_goal`)
  - `"too_broad"` → `generate_questions` → **interrupt** (return questions to the client)
- On next call with the user's answer: `refine_goal` → back to `assess_goal`
- **Safety cap:** if the goal is still too broad after N rounds (e.g. 3), `assess_goal` emits the best-effort `narrowed_goal` and ends, so the user is never stuck in the loop.

**State persisted across the loop (in the checkpointer, keyed by `thread_id = f"{session_id}:clarify"`):**
- `working_goal` — the goal as it has been narrowed so far
- `open_dimensions` — the axes still under-specified (e.g. "depth", "prerequisites", "focus area")
- `round_count` — how many clarification rounds have occurred (drives the safety cap)

**Invocation pattern from FastAPI:**

```python
from langgraph.types import Command
config = graph_config(session_id, "clarify")   # thread_id = f"{session_id}:clarify"

# First call (POST /sessions) — assess the raw goal, then either end with a
# narrowed_goal or interrupt with clarifying questions.
result = clarify_graph.invoke({"goal": raw_goal}, config)
# result = {"narrowed_goal": "..."}  (specific enough → phase probing)
# or result = {"clarifying_questions": [...]}  (paused at interrupt → phase clarifying)

# Subsequent calls (POST /sessions/{id}/clarify) — resume with the user's answer.
result = clarify_graph.invoke(Command(resume={"answer": "..."}), config)
# result = {"narrowed_goal": "..."}  (now specific enough)
# or result = {"clarifying_questions": [...]}  (another round)
```

---

### Graph 2: Probe Graph

**Purpose:** Adaptive questioning loop that brackets the learner's knowledge boundary. Before the loop, the graph decomposes the narrowed goal into its prerequisite strands and seeds the boundary map with them, so the loop works against a **fixed strand set** (rather than inventing strands per question and collapsing to one).

**Entry:** `(narrowed_goal)` — first call; subsequent calls resume with the learner's answer  
**Exit:** `boundary_map` (when done) or `next_question` (when continuing)

```
┌────────────────────────────────────────────┐
│                  PROBE GRAPH               │
│                                            │
│   decompose_strands                        │
│   (one-shot: seed boundary_map             │
│    with the full prerequisite set)         │
│            │                               │
│            ▼                               │
│  ┌─────────────────────┐                   │
│  │ generate_question     │                 │
│  └──────────┬──────────┘                   │
│            ▼                               │
│       [interrupt]                          │
│            │  next_question                │
│            ▼                               │
│  ┌─────────────────────┐                   │
│  │ evaluate_answer       │                 │
│  └──────────┬──────────┘                   │
│            ▼                               │
│  ┌─────────────────────┐                   │
│  │ decide_next           │                 │
│  └───────┬───────┬─────┘                   │
│           │        │                       │
│      continue    done                      │
│           │        │                       │
│           │        ▼                       │
│           │   ┌───────────┐                │
│           └───│   END     │  → boundary_map│
│        (loop   │(bracketed)│               │
│         back)  └───────────┘               │
└────────────────────────────────────────────┘
```

**Nodes:**

| Node | LLM task |
|------|----------|
| `decompose_strands` | *(once, at start)* Given the `narrowed_goal`, enumerate the **direct** prerequisite strands the learner must already understand to learn that specific goal — scoped tightly to the goal itself, not the course/subject it belongs to (no adjacent topics, follow-ups, or prerequisites-of-prerequisites). Seed `boundary_map` with every strand at `{floor: null, ceiling: null, gap_type: "unknown"}` and record the fixed `strands` list. |
| `generate_question` | Given `goal` + `strands` + `strand_descriptions` + `history` + `boundary_map`, produce the next MCQ targeting a strand that is **not yet fully bracketed**. The question tests **only that strand's knowledge** — answerable with the strand's concepts + basic arithmetic alone: never the goal itself, never concepts beyond the strand (a confounded question would mislocate the edge). One question, one concept. Options are built by mutation: correct claim first, distractors = specific misconceptions stated in parallel form (bare claims, no reasoning inside options, no asymmetric emphasis). Binary-search the edge: jump sharply harder after a correct, narrow in after a miss. |
| `evaluate_answer` | Given the question, the learner's answer, and history: mark correct/incorrect, update the **full** `boundary_map` (all seeded strands) for the relevant strand (floor/ceiling/gap_type), append to `history`. A wrong answer is classified `narrow` (isolated slip) vs `systematic` (confidently-held wrong model that must be probed around, not topped up). |
| `decide_next` | Given updated boundary: are **all seeded strands** bracketed (non-null floor AND ceiling, or `gap_type == "none"`)? Have we hit 10 questions? → return `"continue"` or `"done"`. |

**Control flow:**
- `decompose_strands` *(once)* seeds the `strands` set and the `boundary_map`, then flows to `generate_question`
- `generate_question` → **interrupt** (return question to FastAPI handler, which returns it to the client)
- On next call with answer: `evaluate_answer` → `decide_next` → conditional edge:
  - `"continue"` → `generate_question` → interrupt
  - `"done"` → END (output `boundary_map`)

**Invocation pattern from FastAPI:**

The graph is paused/resumed by the **checkpointer** keyed on `thread_id = f"{session_id}:probe"` — the handler does not track the pause itself.

```python
from langgraph.types import Command
config = graph_config(session_id, "probe")   # thread_id = f"{session_id}:probe"

# First call (no answer) — decompose_strands enumerates the goal's
# prerequisite strands (seeding the boundary map), then generate_question
# emits question 1, then the graph pauses at the interrupt.
result = probe_graph.invoke(
    {"goal": session["narrowed_goal"], "history": [], "boundary_map": {},
     "question_count": 0},
    config,
)
# result = {"next_question": {...}}   (graph is now paused; boundary_map is seeded)

# Subsequent calls — resume with the learner's answer. The checkpointer
# restores the paused state and continues from the pending node.
result = probe_graph.invoke(
    Command(resume={"question_id": ..., "selected_index": ...}),
    config,
)
# result = {"next_question": {...}} (paused again) or {"boundary_map": {...}} (done)
```

**State persisted across the loop (in the checkpointer, keyed by `thread_id = f"{session_id}:probe"`):**
- `strands` — the fixed prerequisite-strand set produced by `decompose_strands`
- `strand_descriptions` — one-line description per strand, so `generate_question` knows exactly what each strand covers
- `boundary_map` — seeded with every strand; updated per answer
- `history` — the Q&A transcript so far
- `question_count` — how many questions have been asked (drives the safety cap)
- `next_question` — the current question (output at the interrupt)

> **Why a fixed strand set?** `decide_next` only ends the probe when *every* seeded strand is bracketed. If strands were invented per question (as before), the map would collapse to the single strand just answered and the probe would stop after one question. Seeding the full prerequisite set up front is what makes the loop actually bracket the whole boundary.

---

### Graph 3: Plan Graph

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

As with the Probe graph, the pause is stored in the **checkpointer** (`thread_id = f"{session_id}:plan"`); the `current_plan` and `research` baselines live in the checkpoint, not in the handler.

```python
from langgraph.types import Command
config = graph_config(session_id, "plan")   # thread_id = f"{session_id}:plan"

# Initial generation (first call, no prior plan) — runs research → design → render,
# then pauses at the interrupt (checkpoint saved).
result = plan_graph.invoke(
    {
        "goal": session["narrowed_goal"],
        "boundary_map": session["boundary_map"],
        "response": None,
    },
    config,
)
# result = { "plan": {...} }  (graph is now paused at interrupt)

# User adjusts (subsequent call) — resume; checkpointer restores the paused state
# and the graph re-enters at design_plan (research is skipped).
result = plan_graph.invoke(
    Command(resume={ "action": "adjust", "text": "I don't know what gravity is — add a step…" }),
    config,
)
# result = { "plan": {...} }  (graph interrupts again with refined plan)

# User approves (final call) — resume to the terminal branch.
result = plan_graph.invoke(
    Command(resume={ "action": "approve" }),
    config,
)
# result = { "plan": {...}, "approved": True }  (graph reaches END)
```

**State persisted across the loop (in the checkpointer, keyed by `thread_id = f"{session_id}:plan"`):**
- `current_plan` — the last rendered plan (baseline for next refinement)
- `design_steps` — the current pass's design output (carries the adjustment into the renderer)
- `research` — the topic research output (reused across refinements)
- `pass_count` — how many refinement passes have occurred

These live in the `MemorySaver` checkpoint, so the graph resumes from its in-memory checkpoint between requests within a process run (lost on restart, same as the domain DB for the MVP).

---

### Graph 4: Material Generation Graph

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

### Material generation driver (background task)

Generating slides + questions for every step can take minutes, so `POST /plan/approve` should **not** block. It approves the plan, **returns `202` immediately**, and schedules the per-step loop as a **background task**. The task writes one `StepMaterial` row per step and commits after each — so the **domain DB is the resume point**. The graph checkpointer is deliberately *not* involved, keeping the two stores independent (no double-stored state).

**Flow:**

```
POST /plan/approve
   │
   ├─ write approved Plan row (request db)
   ├─ background_tasks.add_task(generate_materials, session_id, steps, boundary_map)
   └─ return 202 { "phase": "generating", ... }

generate_materials (background task, its OWN db session):
   for step in steps (dependency order):
        if StepMaterial(step) already exists: skip     # resume past it
        result = material_graph.invoke(...)            # linear per-step graph
        db.add(StepMaterial(...)); db.commit()         # one durable row per step
        summaries.append(result["summary"])
```

**Driver code:**

```python
from fastapi import BackgroundTasks

async def generate_materials(session_id: str, steps: list[dict], boundary_map: dict):
    db = SessionFactory()          # its OWN session — never reuse the request's
    try:
        summaries: list[dict] = []
        for step in steps:
            if db.query(StepMaterial).filter_by(
                session_id=session_id, step_id=step["id"]).first():
                continue            # already generated — resume past it
            result = material_graph.invoke(
                {"step": step, "established_concepts": summaries,
                 "learner_context": boundary_map},
            )
            db.add(StepMaterial(
                session_id=session_id, step_id=step["id"],
                slides=result["slides"], questions=result["questions"],
                summary=result["summary"],
            ))
            db.commit()             # durable checkpoint, one step at a time
            summaries.append(result["summary"])
    finally:
        db.close()

@router.post("/sessions/{id}/plan/approve", status_code=202)
async def approve(id: str, background_tasks: BackgroundTasks, db=Depends(get_db)):
    # ... write approved Plan row via `db` ... (commit)
    background_tasks.add_task(generate_materials, id, plan["steps"], boundary_map)
    return {"phase": "generating", "message": "Material generation started."}

@router.get("/sessions/{id}/materials")
async def materials_status(id: str, db=Depends(get_db)):
    session = db.get(Session, id)
    done = db.query(StepMaterial).filter_by(session_id=id).all()
    return {"phase": session.phase, "generated_steps": [m.step_id for m in done]}
```

**Key points:**

| Concern | Handling |
|---------|----------|
| Don't block the request | `POST /plan/approve` returns `202` before generation starts |
| Resume across restarts | Each step is a committed `StepMaterial` row; on retry, skip steps that already exist |
| Don't corrupt the request session | The task opens its **own** DB session (the request's is closed by the time it runs) |
| Keep stores independent | Resume state lives in the **DB**, not the graph checkpointer — no double-stored state |
| Client visibility | Poll `GET /sessions/{id}/materials` until phase flips to `executing` |

**Durability / migration:** `BackgroundTasks` runs **in-process** — no retry, lost on crash/restart. That matches the in-memory MVP (everything is lost on restart anyway). When you move to Postgres and want generation to survive restarts, **swap `add_task` for a queue job** (ARQ / RQ / Celery / Dramatiq / SQS) with a worker — the `generate_materials` body stays identical, only the enqueue call changes.

---

## Graph Invocation Flow (end-to-end)

```
Client                    FastAPI Handler              LangGraph
  │                           │                           │
  │── POST /sessions ────────►│── invoke ───────────────►│  [Clarify Graph: assess_goal]
  │◄── session_id, phase ────│   (narrowed_goal or       │
  │                           │    clarifying_questions)  │
  │                           │                           │
  │── POST /clarify {ans} ───►│── resume ───────────────►│  [Clarify Graph: refine → assess]
  │◄── narrowed_goal ────────│   (or more questions)     │
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
| 1 | **Clarify** | Clarifying | 1-2 (assess + optional questions) per round | Yes — interrupt/resume loop until specific (capped) |
| 2 | **Probe** | Probing | 2 first turn (`decompose_strands` + `generate_question`); 1-2 after (`evaluate_answer` [+ `generate_question`]; `decide_next` is deterministic) | Yes — interrupt/resume loop |
| 3 | **Plan** | Planning, Reviewing | 2-3 (research once + design + render per pass) | Yes — interrupt/resume loop until approved |
| 4 | **Material** | Generating (on approve) | 3 (slides + questions + summary) per step | No — linear, called N times |

Execution (step 6) requires **no graph and no LLM** — it's a DB read + index comparison.

---

## Implementation Notes

1. **Clarify graph uses LangGraph interrupts, held by an in-memory checkpointer.** `POST /sessions` makes the first call (`assess_goal`); if the goal is too broad the graph interrupts with clarifying questions, and each `POST /sessions/{id}/clarify` resumes with `Command(resume={"answer": ...})`. The `working_goal`, `open_dimensions`, and `round_count` live in the checkpointer (`thread_id = f"{session_id}:clarify"`) across rounds. A safety cap (e.g. 3 rounds) forces a best-effort `narrowed_goal` so the user is never stuck. On exit, `narrowed_goal` is written to the `Session` row and the phase advances to `probing`.

2. **Probe graph uses LangGraph interrupts, held by an in-memory checkpointer.** On the first call the `decompose_strands` node enumerates the goal's prerequisite strands and seeds `boundary_map` (each at `{floor: null, ceiling: null, gap_type: "unknown"}`), giving the loop a fixed strand set; `decide_next` ends only when *all seeded* strands are bracketed (or the 10-question cap is hit). The `generate_question` node emits the question and the graph pauses; the pause (pending node + accumulated state) is written to the `MemorySaver` checkpointer under `thread_id = f"{session_id}:probe"`. The next `POST /probe` call resumes by invoking with `Command(resume=answer)` and the same config — the checkpointer restores the paused state, so the handler never re-derives it.

3. **Plan graph uses LangGraph interrupts (same pattern as Probe), held by the in-memory checkpointer.** The graph pauses after `render_plan` (checkpoint saved in memory) and resumes via `Command(resume=...)` when the user's next response arrives (adjust or approve). The `current_plan` and `research` baselines live in the checkpointer (keyed by `thread_id = f"{session_id}:plan"`) across the loop, not in the domain DB. On approval, the final plan is written to the `Plan` row (same `session_id`, incremented `version` on each pass).

4. **Material generation is sequential and must stay sequential.** Each step's `write_slides` node needs the accumulated `ConceptSummary` list from all prior steps, so steps must be generated in order. The context stays small (~1-2k tokens for the summaries) regardless of plan length. For large plans, run it as a background task with a polling endpoint (see the **Material generation driver** subsection) so the client isn't waiting synchronously.

5. **Execution is purely deterministic.** No LLM, no graph. The `POST .../answers` handler compares `selected_index` to `correct_index` for each question, computes the score, and returns results. The "revisit" message on failure is a static template (e.g. "Review the slides before retrying"), not an LLM-generated suggestion.

6. **Both stores are in-memory for the MVP and migrate to Postgres together.** Domain state (sessions, questions, plan, materials, progress) is in an in-memory SQLite DB; graph execution state is in an in-memory `MemorySaver` checkpointer. On process restart both are lost (acceptable for MVP). **Later migration to Postgres:** change the domain engine's connection string to a Postgres URL (the ORM models don't change), and swap the checkpointer from `MemorySaver` to `PostgresSaver` (same interface, only the construction changes). The two stores remain independent: the checkpointer never holds API-queryable domain records, and the domain DB never holds graph execution state.

7. **Use a session-scoped DB session per request.** FastAPI dependency:
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
