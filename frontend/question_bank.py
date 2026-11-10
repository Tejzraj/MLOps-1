"""Load curated practice prompts, separate from training and evaluation data."""

from dataclasses import dataclass
from pathlib import Path

import yaml


@dataclass(frozen=True)
class Question:
    """One practice prompt with its educational reference answer."""

    question_id: str
    topic: str
    question: str
    reference_answer: str


def load_questions(path: Path) -> list[Question]:
    """Load a nonempty bank with unique identifiers and nonblank fields."""
    try:
        rows = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(rows, list) or not rows:
            raise ValueError("question bank must be a nonempty list")
        questions = [Question(**row) for row in rows]
        for question in questions:
            if not all(
                isinstance(value, str) and value.strip() for value in vars(question).values()
            ):
                raise ValueError("question fields must be nonempty strings")
        if len({item.question_id for item in questions}) != len(questions):
            raise ValueError("question identifiers must be unique")
        return questions
    except (OSError, TypeError, ValueError, yaml.YAMLError) as exc:
        raise ValueError(f"Unable to load practice questions: {exc}") from exc
