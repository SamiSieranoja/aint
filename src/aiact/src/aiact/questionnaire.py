from __future__ import annotations
from pathlib import Path
import yaml
from prompt_toolkit import prompt
from prompt_toolkit.formatted_text import HTML
from aiact.models import Question, QuestionChoice, DependsOn, SystemProfile
from aiact import fuzzy

DATA_DIR = Path(__file__).parent.parent.parent / "data"
QUESTIONS_FILE = DATA_DIR / "questions.yaml"

FUZZY_YES_NO_CHOICES = [
    QuestionChoice(value="yes",          label="Yes, definitely"),
    QuestionChoice(value="probably_yes", label="Probably yes"),
    QuestionChoice(value="unknown",      label="Not sure / don't know"),
    QuestionChoice(value="probably_no",  label="Probably not"),
    QuestionChoice(value="no",           label="No"),
]


def load_questions(path: Path = QUESTIONS_FILE) -> list[Question]:
    raw = yaml.safe_load(path.read_text())
    questions = []
    for item in raw["questions"]:
        if item.get("type") == "fuzzy_yes_no":
            item["choices"] = [c.model_dump() for c in FUZZY_YES_NO_CHOICES]
        depends_raw = item.pop("depends_on", None)
        depends = DependsOn(**depends_raw) if depends_raw else None
        questions.append(Question(**item, depends_on=depends))
    return questions


def _should_ask(question: Question, answers: dict[str, str]) -> bool:
    if question.depends_on is None:
        return True
    dep = question.depends_on
    parent_answer = answers.get(dep.question)
    return parent_answer in dep.values


def _ask_question(question: Question) -> str:
    choices = question.choices
    lines = [f"\n<b>{question.text}</b>"]
    if question.help:
        lines.append(f"<i>  ({question.help})</i>")
    for i, c in enumerate(choices, 1):
        lines.append(f"  [{i}] {c.label}")
    print("\n" + "\n".join(lines[0:1]))
    if question.help:
        print(f"  ({question.help})")
    for i, c in enumerate(choices, 1):
        print(f"  [{i}] {c.label}")

    valid = {str(i) for i in range(1, len(choices) + 1)}
    while True:
        try:
            raw = prompt(HTML("<ansicyan>Answer [1-{}]: </ansicyan>".format(len(choices)))).strip()
        except (EOFError, KeyboardInterrupt):
            raise SystemExit(0)
        if raw in valid:
            return choices[int(raw) - 1].value
        print(f"  Please enter a number between 1 and {len(choices)}.")


class QuestionnaireRunner:
    def __init__(self, questions_path: Path = QUESTIONS_FILE) -> None:
        self._questions = load_questions(questions_path)

    def run(self) -> dict[str, str]:
        answers: dict[str, str] = {}
        for q in self._questions:
            if _should_ask(q, answers):
                answers[q.id] = _ask_question(q)
        return answers

    def build_profile(self, answers: dict[str, str]) -> SystemProfile:
        slot_values: dict[str, float] = {}
        for q in self._questions:
            if q.clips_slot and q.id in answers:
                slot_values[q.clips_slot] = fuzzy.map_answer(answers[q.id])
        return SystemProfile(**slot_values)
