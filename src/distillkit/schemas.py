"""Pydantic schemas for teacher/judge replies that must be machine-readable.

Teacher *answers* stay free text on purpose: the student learns whatever format the teacher writes.
"""

import re
from typing import Literal

from pydantic import BaseModel, Field, ValidationError


class GeneratedQuestion(BaseModel):
    question: str = Field(min_length=10, description="The question text only, in the persona's own words")


class JudgeVerdict(BaseModel):
    reasoning: str = Field(description="Short comparison of both answers against the reference")
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
