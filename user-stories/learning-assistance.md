---
name: learning-assistance
description: End-to-end assisted learning flow. The user states a goal, the system probes their current understanding, generates a reviewable plan, produces slide-based learning materials with self-assessment quizzes, and the user executes the plan to mastery.
---

# User Story: Learning Assistance

## Primary Story

**As a** learner,
**I want** to tell the system what I want to learn in a free paragraph and then be guided from my current understanding to mastery through a personalised, reviewable, step-by-step plan,
**so that** I build genuine understanding (not memorisation) at my own pace, with a clear picture of what's ahead before I commit.

---

## Detailed User Journey

### 1. The learner states their goal

- The user writes a free-form paragraph describing what they want to learn (e.g. *"I want to learn Newton's second law of motion."*).
- **If the target is too broad** (e.g. *"I want to learn physics"*), the system asks the user to narrow it before proceeding. This is a clarification loop, not a gate — the system probes with concrete sub-topics until the target is specific enough to plan against.
- The system records the goal as the **learning objective** that every subsequent step targets.

### 2. The system probes the learner's boundary

- The system runs a **short run (max 10 questions)** of multiple-choice / true-false self-assessment questions.
- These questions are **adaptive**: each subsequent question is informed by the previous answers, aiming to locate the *edge* of the learner's understanding (the frontier between what they reliably know and what they don't).
- The probe covers **every prerequisite strand** the planned lesson will depend on — not just the headline topic.
- The output of this phase is a **boundary map**: for each relevant strand, what the learner has (floor) and where it ends (ceiling).
- **All-correct is not "done"** — it means the questions were too easy. The system escalates difficulty until the edge is bracketed.
- **One wrong answer is not "done" either** — the system probes around it to characterise the gap (careless slip vs. narrow gap vs. systematic misconception) before concluding.

### 3. The system generates a plan

- Using the learning objective (step 1) and the boundary map (step 2), the system generates an **ordered list of incrementally small learning steps** from the learner's current boundary to mastery.
- **Dependencies come first**: foundational concepts are sequenced before the concepts that build on them (e.g. understand `F`, `m`, `a` individually before combining into `F = ma`).
- Each step is small enough to be a single focused learning unit.
- The plan is presented as:
  1. **A prose summary** — what will be covered, in what order, and why this order given the learner's starting point and goal.
  2. **A dependency map** — a small directed acyclic graph (DAG) showing the prerequisite relationships between steps (e.g. rendered as Mermaid).
- The plan is **not yet locked in** — it goes to step 4.

### 4. The learner reviews and adjusts the plan

- The learner can, **in their own words**, add, remove, or modify steps, and adjust the depth or difficulty of any step (e.g. *"I don't know the definition of gravity — add this"*, or *"skip the vector algebra, I'm comfortable with that"*).
- **Every adjustment regenerates the full plan** — the system re-derives the ordered sequence with dependencies, rather than patching a single node. This ensures the dependency graph stays consistent.
- The learner iterates (adjust → regenerate → review) until they are satisfied.
- **Execution starts only when the learner explicitly approves the plan.** There is no implicit "auto-start."

### 5. The system generates detailed learning materials

For each step in the approved plan, the system generates:

- **A set of learning slides**, formatted as **HTML**, that teach the step's content. Each step contains several slides (the exact count is determined by the depth of the material). The slides follow the pedagogical principles in `ref/teacher.md`:
  - Unconditional truths are established first and confirmed before building on them.
  - Every concept is motivated — the learner is shown *why* it exists and *how* it could have been discovered, not just *what* it is.
  - Dependencies are made explicit: each new concept is connected to what came before it.
- **A set of self-assessment questions** (multiple-choice / true-false) that follow the slides. These questions:
  - Test that the step's concepts actually landed (not just recognition).
  - Are written so that distractors are plausible real misconceptions, not filler.
  - Have correct answers and explanations (explanations revealed after answering).
  - Cover the step's key concepts at varying depths.
- The materials for step *N+1* may reference and build on the concepts established in step *N*, but each step's materials are self-contained enough to be revisited independently.

### 6. The learner executes the plan

- The learner works through the plan **step by step, in order**.
- For each step:
  1. **Learn** — read/study the HTML slides.
  2. **Assess** — answer the ensuing self-assessment questions.
- A step is considered **complete** when the learner has finished all its slides **and** all its questions.
- The system tracks progress: which steps are done, in-progress, or pending.
- If the learner misses questions, the system can flag which specific concepts need re-study (revisit the relevant slides) before moving to the next step.
- Once all steps are complete, the learner has reached the **mastery state** defined by the original learning objective.

---

## Acceptance Criteria

### Goal capture
- [ ] User can enter a free-form paragraph as the learning goal.
- [ ] If the goal is too broad, the system asks clarifying questions to narrow it before probing.
- [ ] The narrowed goal is stored and used as the target for the entire session.

### Boundary probe
- [ ] The system generates at most 10 multiple-choice / true-false questions.
- [ ] Questions are adaptive: later questions depend on earlier answers.
- [ ] The probe covers all prerequisite strands relevant to the goal, not just the headline topic.
- [ ] If the learner answers all questions correctly, the system escalates difficulty (does not stop at 10 if the edge hasn't been bracketed — unless 10 is exhausted).
- [ ] If the learner misses a question, the system probes around it to characterise the gap before concluding.
- [ ] The output is a structured boundary map (per-strand floor and ceiling).

### Plan generation
- [ ] The system produces an ordered list of small, incrementally-sized learning steps.
- [ ] Steps are sequenced by dependency (prerequisites before dependents).
- [ ] The plan includes a prose summary explaining the order and rationale.
- [ ] The plan includes a dependency DAG (e.g. Mermaid graph).
- [ ] The plan starts from the learner's current boundary (step 2 output), not from zero and not from the goal.

### Plan review & adjustment
- [ ] The learner can add, remove, or modify steps using free-form text.
- [ ] The learner can adjust the depth or difficulty of any step.
- [ ] Every adjustment triggers a full plan regeneration (not a partial patch).
- [ ] The learner can iterate multiple times (adjust → regenerate → review).
- [ ] Execution does not start until the learner explicitly approves.

### Material generation
- [ ] Each step has a set of HTML-formatted learning slides.
- [ ] Each step has a set of multiple-choice / true-false self-assessment questions.
- [ ] Questions include correct answers and explanations (explanations shown post-answer).
- [ ] Slides follow the pedagogical principles: unconditional truths first, motivated discovery, explicit dependency connections.
- [ ] Materials are generated only after the plan is approved (step 4).

### Execution
- [ ] The learner works through steps in the planned order.
- [ ] Each step requires completing all slides and all questions.
- [ ] Progress is tracked (done / in-progress / pending per step).
- [ ] Missed questions trigger a suggestion to revisit relevant slides before proceeding.
- [ ] Completion of all steps = mastery of the learning objective.

---

## Out of Scope (for now)

- Multi-session / resumable learning across days (the flow above is a single continuous session).
- Social features (sharing plans, peer review, leaderboards).
- Content sourced from external APIs / textbooks (materials are AI-generated).
- Non-multiple-choice assessment formats (open-ended essays, code challenges, etc.).
- Real-time instructor interaction (the system is autonomous; the learner interacts only with the AI).

---

## References

- `ref/teacher.md` — pedagogical principles (unconditional truths first, motivated discovery, dependency graph, quiz construction rules) that govern how slides and questions are authored in step 5.
