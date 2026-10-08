"""Pydantic schemas for teacher/judge replies that must be machine-readable.

Teacher *answers* stay free text on purpose: the student learns whatever format the teacher writes.
"""

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError


class GeneratedQuestion(BaseModel):
    model_config = ConfigDict(extra='forbid')
    question: str = Field(min_length=10, description="The question text only, in the persona's own words")


QUESTION_TRANSPORT_VERSION = 'question-json-v2-client-length-validation'


def question_transport_schema() -> dict:
    """Keep semantic length validation in Pydantic, not the decoder grammar.

    Installed xgrammar 0.2.7 compiles minLength strings to a character rule that
    forbids JSON escapes, including newlines and quotes needed in code questions.
    This known single-field transport relaxation never changes parse_json checks.
    """
    schema = GeneratedQuestion.model_json_schema()
    schema['properties']['question'].pop('minLength')
    return schema


class JudgeVerdict(BaseModel):
    verdict: Literal["A", "B", "T"] = Field(description="A or B for the better answer, T for a tie")


_FENCE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL)


def parse_json[M: BaseModel](model: type[M], text: str) -> M | None:
    """Validate a JSON reply against `model`; tolerates a ```json fence. None if invalid."""
    text = text.strip()
    if m := _FENCE.match(text):
        text = m.group(1)
    try:
        return model.model_validate_json(text)
    except ValidationError:
        return None
