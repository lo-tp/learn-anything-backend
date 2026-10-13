r"""Tests for core/model_answers.py (#195).

The class of failure these describe: a model asked to answer in JSON writes a
backslash — `\int`, `\frac`, `\beta` — that JSON does not accept as an escape,
or accepts as a control character the model never meant. This module is what
reads such an answer, and what it does when the answer cannot be read.
"""

from __future__ import annotations

import json
from typing import Annotated

import httpx2  # the httpx the openai SDK 3.x builds its errors from
import openai
import pytest
from pydantic import BaseModel, Field, ValidationError

from core.model_answers import (
    ModelAnswerUnparseable,
    ModelRecovery,
    backend_refused_json,
    drain_model_recoveries,
    json_shape_spec,
    read_json_answer,
    record_model_recovery,
    repair_json_escapes,
)
from graphs.clarify.schemas import AssessOut
from graphs.probe.schemas import BatchQuestionsOut
from tests.error_shapes import (
    INVALID_ESCAPE,
    api_error,
    model_unavailable,
    structured_output_refused,
)

# --- repair_json_escapes ---


class TestRepairJsonEscapes:
    """The repair is one pass over the text: legal escape pairs are consumed as
    pairs, so a string that was already correct is never made worse."""

    def test_the_backslash_the_model_could_not_write_is_doubled(self):
        raw = '{"key_points": ["$\\int_a^b f(x)\\,dx$"]}'

        repaired, count = repair_json_escapes(raw)

        assert repaired == '{"key_points": ["$\\\\int_a^b f(x)\\\\,dx$"]}'
        assert count == 2

    def test_a_legal_escape_pair_is_left_alone(self):
        """`\\n` is a newline in JSON, `\\u00e9` is an é, `\\\\` is a backslash.

        Doubling any of them would rewrite text the model meant literally.
        """
        raw = '{"text": "line one\\nnext\ttab\\u00e9 C:\\\\path say\\"hi\\""}'

        repaired, count = repair_json_escapes(raw)

        assert repaired == raw
        assert count == 0

    def test_repairing_already_valid_text_changes_nothing(self):
        """An idempotent repair is what makes a retry safe."""
        valid = json.dumps({"key_points": ["$\\frac{1}{2}$"]})

        repaired, count = repair_json_escapes(valid)

        assert repaired == valid
        assert count == 0

    def test_form_feed_and_backspace_are_read_as_macros(self):
        r"""`\f` and `\b` are legal JSON escapes — and they are what `\frac`,
        `\beta` and `\binom` become when JSON reads them.

        A real form feed or backspace is not a thing this app's text contains,
        so reading them as the start of a macro is the reading that preserves
        what the model wrote. `$\frac{1}{2}$` must stay a fraction, not arrive
        as a control character plus `rac{1}{2}`.
        """
        raw = '{"key_points": ["$\\frac{1}{2}$", "\\beta decay", "\\binom{n}{k}"]}'

        repaired, count = repair_json_escapes(raw)

        parsed = json.loads(repaired)
        assert parsed["key_points"] == [
            "$\\frac{1}{2}$",
            "\\beta decay",
            "\\binom{n}{k}",
        ]
        assert count == 3

    def test_a_lone_trailing_backslash_is_doubled(self):
        # Built by concatenation: a raw string cannot end in a lone backslash.
        raw = '{"text": "a trailing backslash' + "\\"

        repaired, count = repair_json_escapes(raw)

        assert repaired == '{"text": "a trailing backslash' + "\\\\"
        assert count == 1

    def test_the_count_is_what_makes_the_repair_visible(self):
        """An uncounted repair is a silent one: #195 is about a failure nobody
        could see, so the number is part of the fix, not a detail."""
        _, count = repair_json_escapes('{"a": "$\\sum_1^n \\frac{1}{n}$"}')

        assert count == 2


# --- json_answer ---


class TestReadJsonAnswer:
    def test_reads_the_object_a_model_answered_with(self):
        answer = read_json_answer('{"verdict": "specific", "narrowed_goal": "x"}')

        assert answer.value["verdict"] == "specific"
        assert answer.repaired_escapes == 0

    def test_reads_an_object_out_of_the_fencing_a_model_adds(self):
        text = "```json\n{\"verdict\": \"too_broad\"}\n```"

        assert read_json_answer(text).value == {"verdict": "too_broad"}

    def test_reads_an_object_with_prose_around_it(self):
        text = 'Here is the plan:\n{"verdict": "specific"}\nHope that helps!'

        assert read_json_answer(text).value == {"verdict": "specific"}

    def test_the_latex_survives_the_round_trip(self):
        """The end this exists for: a key point the model wrote in LaTeX arrives
        as the LaTeX, with no control characters in it."""
        text = '{"key_points": ["$\\int_a^b f(x)\\,dx$ is area", "\\beta = 1/√(1-v²)"]}'

        answer = read_json_answer(text)

        assert answer.value["key_points"][0] == "$\\int_a^b f(x)\\,dx$ is area"
        assert answer.value["key_points"][1] == "\\beta = 1/\u221a(1-v\u00b2)"
        assert answer.repaired_escapes == 3
        assert "\u000c" not in json.dumps(answer.value)

    def test_a_brace_inside_a_string_does_not_cut_the_answer_short(self):
        text = '{"steps": [{"description": "a {nested} brace, then \\"quote\\""}]}'

        assert read_json_answer(text).value["steps"][0]["description"].startswith("a {nested}")

    def test_the_repair_is_reported_because_the_caller_has_to_say_what_happened(self):
        answer = read_json_answer('{"text": "$\\sum_1^n$"}')

        assert answer.repaired_escapes == 1

    def test_an_answer_that_is_not_json_is_said_so(self):
        with pytest.raises(ValueError, match="no JSON object"):
            read_json_answer("I could not decide.")

    def test_a_truncated_answer_is_said_so(self):
        with pytest.raises(ValueError, match="no JSON object"):
            read_json_answer('{"key_points": ["unfinished')

    def test_a_malformed_unicode_escape_is_still_a_failure(self):
        """Doubling what JSON rejects is the repair. A backslash-u with bad hex
        is a different fault, and must not be quietly rewritten."""
        with pytest.raises(ValueError, match="not valid JSON"):
            read_json_answer('{"text": "\\uZZZZ"}')


# --- json_shape_spec ---


class TestJsonShapeSpec:
    """The spec the second plan gives the model: the shape of an *instance*, not
    the schema — the one sample of an unconstrained ask on this backend had the
    model answering with the schema itself."""

    def test_it_states_the_fields_of_the_answer(self):
        spec = json_shape_spec(AssessOut)

        assert "narrowed_goal" in spec
        assert "open_dimensions" in spec

    def test_it_is_not_the_schema_dumped(self):
        """A model shown a JSON Schema tends to return the schema back."""
        spec = json_shape_spec(BatchQuestionsOut)

        assert "additionalProperties" not in spec
        assert "$defs" not in spec

    def test_a_nested_model_is_shown_as_the_object_it_produces(self):
        spec = json_shape_spec(BatchQuestionsOut)

        assert "correct_index" in spec
        assert "options" in spec

    def test_a_literal_field_shows_which_values_are_allowed(self):
        spec = json_shape_spec(AssessOut)

        assert "specific" in spec
        assert "too_broad" in spec

    def test_the_spec_is_itself_an_instance_of_the_shape(self):
        """So the model can copy it: the placeholders stand in a JSON document."""
        spec = json_shape_spec(BatchQuestionsOut)

        assert spec.strip().startswith("{")
        assert spec.strip().endswith("}")

    def test_a_field_the_prompt_only_asks_for_optionally_is_still_shown(self):
        """Optional in the schema, expected by the prompt: the spec states it, with
        the value its real type asks for."""

        class Part(BaseModel):
            required: str
            maybe: str | None = None
            count: Annotated[int, Field(ge=1, le=5)] = 1

        spec = json_shape_spec(Part)

        assert '"required": "<string>"' in spec
        assert '"maybe": "<string>"' in spec
        assert '"count": 0' in spec

    def test_a_self_referring_model_does_not_recurse_forever(self):
        class Node(BaseModel):
            name: str
            children: list[Node] = Field(default_factory=list)

        spec = json_shape_spec(Node)

        assert "children" in spec


# --- backend_refused_json ---


class TestBackendRefusedJson:
    def test_the_refusal_this_backend_answers_with_is_recognised(self):
        assert backend_refused_json(structured_output_refused()) is True

    def test_the_same_class_named_only_in_the_message_is_recognised(self):
        """The discriminator lives in a body this backend has shaped two ways;
        the message is the one that always describes the fault."""
        exc = api_error(
            status=400, error={"message": INVALID_ESCAPE}, error_cls=openai.BadRequestError
        )

        assert backend_refused_json(exc) is True

    def test_an_unrelated_server_fault_is_not(self):
        """A model that is overloaded is not a model whose JSON did not parse:
        repeating the request is the right plan for that one, and this must not
        send it down the other path."""
        exc = model_unavailable()

        assert backend_refused_json(exc) is False

    def test_a_connection_fault_is_not(self):
        exc = openai.APIConnectionError(request=httpx2.Request("POST", "http://backend/v1"))

        assert backend_refused_json(exc) is False

    def test_an_answer_that_validated_badly_is_not(self):
        """A shape miss is a prompt problem, not a parse problem. Re-asking
        without the schema is a guess at a prompt problem; this says no."""
        try:
            AssessOut.model_validate({"verdict": "unsure"})
        except ValidationError as exc:
            assert backend_refused_json(exc) is False

    def test_an_unrelated_error_type_is_not(self):
        assert backend_refused_json(RuntimeError("model down")) is False


# --- ModelAnswerUnparseable ---


class TestModelAnswerUnparseable:
    def test_it_names_the_stage_the_fault_and_what_it_saw(self):
        exc = ModelAnswerUnparseable(
            stage="material.plan_slide_contents",
            reason="the answer was not JSON",
            raw='{"slide_contents": [',
        )

        assert "material.plan_slide_contents" in str(exc)
        assert "the answer was not JSON" in str(exc)
        assert '{"slide_contents": [' in exc.raw

    def test_a_refused_answer_that_could_not_be_recovered_keeps_its_cause(self):
        original = structured_output_refused()
        try:
            raise ModelAnswerUnparseable("plan.design_plan", "refused", "") from original
        except ModelAnswerUnparseable as exc:
            assert exc.__cause__ is original


# --- recovery records ---


class TestModelRecoveryRecords:
    """A recovery is only useful if it can be attributed. The record drains into
    the stage row that is written next, so "which stage leaned on this" is a
    query and not a guess from log lines."""

    def test_a_recorded_recovery_is_attributed_to_the_next_drain(self):
        record_model_recovery(ModelRecovery(reason="backend_refused_json", repaired_escapes=21))

        records = drain_model_recoveries()

        assert records == [{"reason": "backend_refused_json", "repaired_escapes": 21}]

    def test_draining_clears_so_a_later_stage_is_not_blamed(self):
        record_model_recovery(ModelRecovery(reason="backend_refused_json", repaired_escapes=3))
        drain_model_recoveries()

        assert drain_model_recoveries() == []

    def test_a_recovered_answer_is_also_written_down_as_data(self):
        recovery = ModelRecovery(reason="backend_refused_json", repaired_escapes=4)

        assert recovery.as_dict() == {"reason": "backend_refused_json", "repaired_escapes": 4}
