"""
Generate synthetic dashboard logs with realistic distributions.

The output goes to unified_project/synthetic_logs and matches the same JSON
schema as normal session logs, but contains no real user data.
"""

from __future__ import annotations

import json
import math
import random
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
OUT_DIR = BASE_DIR / "synthetic_logs"
RNG = random.Random(4940)

PHASE_EMOTION = "emotional_discovery"
PHASE_ARTISTIC = "artistic_discovery"
PHASE_CODE = "code_generation"

QUESTION_TEXT = {
    PHASE_EMOTION: {
        "comfort_describing_emotions": "How comfortable are you describing your emotions to the AI?",
        "ai_understands_emotional_tone": "Does the AI understand the emotional tone you want?",
    },
    PHASE_ARTISTIC: {
        "confidence_describing_visual_style": "How confident are you in describing the visual style you want?",
        "familiarity_with_art_terms": "How familiar are you with art or design terms?",
        "knows_how_to_describe_look": "Do you know how to describe the look you want?",
    },
    PHASE_CODE: {
        "output_makes_uncomfortable": "Does this output make you uncomfortable?",
        "satisfaction_with_output": "How satisfied are you with this output?",
        "output_represents_people_fairly": "Does this output represent people fairly?",
        "needs_more_guidance_examples": "Would more guidance or examples help before you revise your prompt?",
    },
}

PHASE_LABELS = {
    PHASE_EMOTION: "Emotion stage",
    PHASE_ARTISTIC: "Artistic stage",
    PHASE_CODE: "Code generation stage",
}

SAFETY_SCAN_VERSION = "local_keyword_v1"
SAFETY_BLOCK_MESSAGE = (
    "I can help work with difficult feelings, but I cannot continue with threats, graphic harm, "
    "or abusive extreme language. Please rephrase the idea in safer, non-violent terms so we can "
    "keep making the sketch."
)
SAFETY_OUTPUT_BLOCK_MESSAGE = (
    "I held back that generated result because it included unsafe visual content. "
    "Please try a safer artistic direction."
)
SAFETY_CHECKS = [
    "input_prompt",
    "output_response",
    "output_emotion_profile",
    "output_artistic_profile",
    "output_artistic_options",
    "generated_code",
    "generated_graph",
]
SAFETY_CATEGORY_LABELS = {
    "abusive_extreme_language": "abusive extreme language",
    "extreme_emotion": "extreme emotion wording",
    "extreme_language": "strong language",
    "graphic_violence": "graphic violence",
    "hate_or_harassment": "hate or harassment",
    "inappropriate_sexual_content": "explicit sexual content",
    "pii": "personally identifiable or secret data",
    "prompt_injection": "prompt injection or instruction override",
    "self_harm": "self-harm wording",
    "violence_reference": "violent reference",
    "violent_element": "violent element or weapon imagery",
    "violent_threat": "violent threat or intent",
    "weapon_threat": "weapon threat",
}
INPUT_BLOCK_CATEGORIES = [
    "prompt_injection",
    "self_harm",
    "violent_threat",
    "weapon_threat",
    "pii",
    "hate_or_harassment",
    "abusive_extreme_language",
]
INPUT_FLAG_CATEGORIES = [
    "extreme_emotion",
    "violence_reference",
    "extreme_language",
]
OUTPUT_BLOCK_CATEGORIES = [
    "violent_element",
    "graphic_violence",
    "inappropriate_sexual_content",
]
OUTPUT_FLAG_CATEGORIES = [
    "extreme_emotion",
    "violence_reference",
    "extreme_language",
]

AGE_WEIGHTS = [
    ("Under 18", 0.06),
    ("18-24", 0.42),
    ("25-34", 0.22),
    ("35-44", 0.13),
    ("45-54", 0.08),
    ("55-64", 0.05),
    ("65+", 0.04),
]
GENDER_WEIGHTS = [("female", 0.42), ("male", 0.36), ("other", 0.12), (None, 0.10)]


def clamp_int(value: float, low: int, high: int) -> int:
    return max(low, min(high, int(round(value))))


def weighted_choice(items: list[tuple[object, float]]):
    total = sum(weight for _, weight in items)
    pick = RNG.random() * total
    running = 0.0
    for value, weight in items:
        running += weight
        if pick <= running:
            return value
    return items[-1][0]


def normal_score(mean: float, sd: float, low: int = 0, high: int = 10) -> int:
    return clamp_int(RNG.gauss(mean, sd), low, high)


def correlated_score(base: int, offset: float, sd: float, low: int = 0, high: int = 10) -> int:
    return clamp_int(base + offset + RNG.gauss(0, sd), low, high)


def maybe_skip(value: int, probability: float):
    return None if RNG.random() < probability else value


def phase_feedback(phase: str, submitted_at: datetime, answers: dict[str, int | None]) -> dict:
    skipped = {key: value is None for key, value in answers.items()}
    revision = {
        "submitted_at": submitted_at.isoformat(),
        "answers": answers,
        "skipped": skipped,
    }
    return {
        "phase": phase,
        "phase_label": PHASE_LABELS[phase],
        "updated_at": submitted_at.isoformat(),
        "scale": "0-10",
        "questions": QUESTION_TEXT[phase],
        "answers": answers,
        "skipped": skipped,
        "revisions": [revision],
    }


def make_survey(started_at: datetime, profile: str) -> dict | None:
    if RNG.random() < 0.08:
        return None

    if profile == "low_tech":
        ai_experience = normal_score(2.2, 1.2)
    elif profile == "high_tech_low_art":
        ai_experience = normal_score(7.4, 1.4)
    elif profile == "confident":
        ai_experience = normal_score(7.8, 1.5)
    else:
        ai_experience = normal_score(5.2, 2.2)

    age = weighted_choice(AGE_WEIGHTS)
    gender = weighted_choice(GENDER_WEIGHTS)
    if RNG.random() < 0.06:
        age = None
    if RNG.random() < 0.05:
        ai_experience = None

    return {
        "type": "demographic_fairness",
        "submitted_at": (started_at + timedelta(seconds=RNG.randint(8, 95))).isoformat(),
        "age_range": age,
        "gender": gender,
        "ai_experience": ai_experience,
        "skipped": {
            "age_range": age is None,
            "gender": gender is None,
            "ai_experience": ai_experience is None,
        },
        "privacy_notes": [
            "Synthetic data for dashboard testing",
            "Contains no real user data",
        ],
    }


def make_scores(profile: str, ai_experience: int | None) -> dict[str, int]:
    tech_anchor = ai_experience if ai_experience is not None else normal_score(5, 2.5)
    if profile == "low_tech":
        comfort = normal_score(3.1, 1.5)
        visual_conf = normal_score(4.0, 1.7)
        art_terms = normal_score(3.7, 1.6)
        describe = normal_score(4.1, 1.6)
        eval_avg = max(0, min(5, RNG.gauss(2.2, 0.9)))
    elif profile == "high_tech_low_art":
        comfort = normal_score(6.3, 1.6)
        visual_conf = normal_score(2.9, 1.3)
        art_terms = normal_score(2.5, 1.2)
        describe = normal_score(3.1, 1.4)
        eval_avg = max(0, min(5, RNG.gauss(3.0, 0.8)))
    elif profile == "emotion_uncertain":
        comfort = normal_score(2.4, 1.2)
        visual_conf = normal_score(5.6, 1.8)
        art_terms = normal_score(5.1, 1.8)
        describe = normal_score(5.0, 1.7)
        eval_avg = max(0, min(5, RNG.gauss(2.5, 0.9)))
    elif profile == "confident":
        comfort = normal_score(7.6, 1.3)
        visual_conf = normal_score(7.3, 1.4)
        art_terms = normal_score(6.9, 1.5)
        describe = normal_score(7.4, 1.3)
        eval_avg = max(0, min(5, RNG.gauss(4.0, 0.6)))
    else:
        comfort = normal_score(5.4, 2.0)
        visual_conf = normal_score(5.3, 2.0)
        art_terms = normal_score(5.0, 2.0)
        describe = normal_score(5.4, 1.9)
        eval_avg = max(0, min(5, RNG.gauss(3.2, 0.8)))

    support_need = max(0, 4 - min(tech_anchor, comfort, visual_conf, art_terms, describe)) / 4
    tone = correlated_score(comfort, 1.2 - support_need * 2.0, 1.4)
    satisfaction = correlated_score(min(tone, visual_conf, describe), 1.1 - support_need * 1.2, 1.3)
    fairness = correlated_score(min(satisfaction, tone), 0.8 - support_need * 0.7, 1.3)
    discomfort = clamp_int(10 - fairness + RNG.gauss(-1.7 + support_need * 2.2, 1.2), 0, 10)
    guidance = clamp_int(10 - min(tech_anchor, art_terms, describe) + RNG.gauss(0.5, 1.3), 0, 10)

    return {
        "comfort_describing_emotions": comfort,
        "ai_understands_emotional_tone": tone,
        "confidence_describing_visual_style": visual_conf,
        "familiarity_with_art_terms": art_terms,
        "knows_how_to_describe_look": describe,
        "output_makes_uncomfortable": discomfort,
        "satisfaction_with_output": satisfaction,
        "output_represents_people_fairly": fairness,
        "needs_more_guidance_examples": guidance,
        "eval_avg": int(round(eval_avg)),
    }


def make_turns(started_at: datetime, scores: dict[str, int], profile: str) -> list[dict]:
    support_need = max(
        0,
        4
        - min(
            scores["comfort_describing_emotions"],
            scores["confidence_describing_visual_style"],
            scores["familiarity_with_art_terms"],
            scores["knows_how_to_describe_look"],
        ),
    )
    base_turns = 2 + RNG.randint(0, 2)
    extra_turns = int(max(0, support_need) * RNG.uniform(0.5, 1.2))
    if scores["needs_more_guidance_examples"] >= 7:
        extra_turns += RNG.randint(1, 3)
    turn_count = max(1, min(10, base_turns + extra_turns + (1 if profile == "confident" and RNG.random() < 0.35 else 0)))

    phases = [PHASE_EMOTION]
    if turn_count > 1:
        phases.append(PHASE_ARTISTIC)
    if turn_count > 2:
        phases.extend([PHASE_CODE] * (turn_count - 2))

    turns = []
    for index, phase in enumerate(phases, start=1):
        score = clamp_int(RNG.gauss(scores["eval_avg"], 0.75), 0, 5)
        timestamp = started_at + timedelta(minutes=index * RNG.randint(1, 5), seconds=RNG.randint(0, 50))
        safety, message_override, response_override = make_turn_safety(phase, profile, scores, index)
        evaluation = None if safety.get("blocked") else {
            "score": score,
            "explanation": evaluation_text(score),
        }
        turns.append(
            {
                "turn_id": index,
                "timestamp": timestamp.isoformat(),
                "phase": phase,
                "user_message": message_override or synthetic_message(phase, profile, index),
                "has_image": RNG.random() < 0.12,
                "has_audio": RNG.random() < 0.04,
                "ai_response": response_override or synthetic_ai_response(phase),
                "safety": safety,
                "evaluation": evaluation,
            }
        )
    return turns


def synthetic_message(phase: str, profile: str, index: int) -> str:
    if phase == PHASE_EMOTION:
        samples = [
            "I want to show a heavy anxious feeling, like everything is pressing inward.",
            "I'm excited but also nervous, like bright weather after being stuck inside.",
            "I don't know the right words, but it feels quiet and stuck.",
            "Can you help me turn this memory into a visual mood?",
        ]
    elif phase == PHASE_ARTISTIC:
        samples = [
            "Maybe it should be layered with soft motion and not too literal.",
            "I want sharp contrast and repeating shapes, but I need help naming the style.",
            "Use colors that feel tense but still hopeful.",
            "Could we compare a few visual directions before coding?",
        ]
    else:
        samples = [
            "Generate the sketch and keep it beginner friendly.",
            "This is close, but it needs calmer movement and less clutter.",
            "The output feels a little off. Can we make the figures more abstract?",
            "I need more examples before I know what to change.",
        ]
    return f"{RNG.choice(samples)} ({profile} synthetic turn {index})"


def synthetic_ai_response(phase: str) -> str:
    if phase == PHASE_EMOTION:
        return "I reflected the emotional tone and asked one focused follow-up."
    if phase == PHASE_ARTISTIC:
        return "I translated the emotion into visual directions and asked the user to choose."
    return "I generated or revised a p5.js sketch and invited targeted feedback."


def evaluation_text(score: int) -> str:
    if score >= 4:
        return "Specific, reflective response with clear details and strong collaboration."
    if score == 3:
        return "Clear response with some useful detail and reasonable engagement."
    if score == 2:
        return "Basic response with limited detail, but still related to the task."
    return "Very brief or vague response that may need more scaffolding."


def support_need_from_scores(scores: dict[str, int]) -> float:
    lowest_support_score = min(
        scores["comfort_describing_emotions"],
        scores["confidence_describing_visual_style"],
        scores["familiarity_with_art_terms"],
        scores["knows_how_to_describe_look"],
    )
    return max(0.0, min(1.0, (4 - lowest_support_score) / 4))


def empty_safety_check(label: str) -> dict:
    return {
        "label": label,
        "scanner": SAFETY_SCAN_VERSION,
        "status": "safe",
        "flagged": False,
        "blocked": False,
        "categories": [],
        "signals": [],
    }


def detected_safety_check(label: str, category: str, blocked: bool) -> dict:
    status = "blocked" if blocked else "flagged"
    return {
        "label": label,
        "scanner": SAFETY_SCAN_VERSION,
        "status": status,
        "flagged": True,
        "blocked": blocked,
        "categories": [category],
        "signals": [
            {
                "category": category,
                "severity": status,
                "reason": SAFETY_CATEGORY_LABELS.get(category, category.replace("_", " ")),
                "excerpt": f"[synthetic excerpt: {category.replace('_', ' ')}]",
            }
        ],
    }


def merge_safety_checks(checks: dict[str, dict], *, hook: str, generated_output_blocked: bool = False) -> dict:
    categories: list[str] = []
    blocked = generated_output_blocked
    flagged = generated_output_blocked
    for check in checks.values():
        blocked = blocked or bool(check.get("blocked"))
        flagged = flagged or bool(check.get("flagged"))
        for category in check.get("categories", []):
            if category not in categories:
                categories.append(category)
    safety = dict(checks)
    safety.update(
        {
            "blocked": blocked,
            "flagged": flagged or blocked,
            "categories": categories,
            "status": "blocked" if blocked else "flagged" if flagged else "safe",
            "middleware": {
                "name": "AppGuardrailMiddleware",
                "hook": hook,
                "blocked": blocked,
            },
        }
    )
    if generated_output_blocked:
        safety["generated_output_blocked"] = True
    return safety


def synthetic_safety_message(category: str, profile: str, index: int, blocked: bool) -> str:
    if blocked:
        return (
            f"This synthetic turn contains a blocked {category.replace('_', ' ')} signal "
            f"for guardrail testing. ({profile} synthetic turn {index})"
        )
    if category == "extreme_emotion":
        return f"I feel overwhelmed and panicky, and I need help turning that intensity into something safer. ({profile} synthetic turn {index})"
    if category == "violence_reference":
        return f"I keep describing the memory with harsh conflict metaphors, but I want a non-graphic abstraction. ({profile} synthetic turn {index})"
    return f"I used unusually strong wording while asking for a revision, but the request is still about the art. ({profile} synthetic turn {index})"


def make_turn_safety(phase: str, profile: str, scores: dict[str, int], index: int) -> tuple[dict, str | None, str | None]:
    checks = {label: empty_safety_check(label) for label in SAFETY_CHECKS}
    support_need = support_need_from_scores(scores)
    input_block_probability = 0.015 + support_need * 0.012
    input_flag_probability = 0.035 + support_need * 0.035
    output_block_probability = 0.014 if phase == PHASE_CODE else 0.003
    output_flag_probability = 0.020 if phase == PHASE_CODE else 0.006
    roll = RNG.random()

    if roll < input_block_probability:
        category = RNG.choice(INPUT_BLOCK_CATEGORIES)
        checks["input_prompt"] = detected_safety_check("input_prompt", category, True)
        return (
            merge_safety_checks(checks, hook="before_agent"),
            synthetic_safety_message(category, profile, index, True),
            SAFETY_BLOCK_MESSAGE,
        )

    roll -= input_block_probability
    if roll < input_flag_probability:
        category = RNG.choice(INPUT_FLAG_CATEGORIES)
        checks["input_prompt"] = detected_safety_check("input_prompt", category, False)
        return (
            merge_safety_checks(checks, hook="after_agent"),
            synthetic_safety_message(category, profile, index, False),
            None,
        )

    roll -= input_flag_probability
    if roll < output_block_probability:
        category = RNG.choice(OUTPUT_BLOCK_CATEGORIES)
        check_key = RNG.choice(["output_artistic_profile", "output_artistic_options", "generated_code"])
        checks[check_key] = detected_safety_check(check_key, category, False)
        return (
            merge_safety_checks(checks, hook="after_agent", generated_output_blocked=True),
            None,
            SAFETY_OUTPUT_BLOCK_MESSAGE,
        )

    roll -= output_block_probability
    if roll < output_flag_probability:
        category = RNG.choice(OUTPUT_FLAG_CATEGORIES)
        check_key = RNG.choice(["output_response", "output_emotion_profile", "generated_graph"])
        checks[check_key] = detected_safety_check(check_key, category, False)
        return merge_safety_checks(checks, hook="after_agent"), None, None

    return merge_safety_checks(checks, hook="after_agent"), None, None


def make_stage_feedback(started_at: datetime, scores: dict[str, int]) -> dict:
    stage_feedback = {}
    if RNG.random() < 0.86:
        answers = {
            "comfort_describing_emotions": maybe_skip(scores["comfort_describing_emotions"], 0.05),
            "ai_understands_emotional_tone": maybe_skip(scores["ai_understands_emotional_tone"], 0.08),
        }
        stage_feedback[PHASE_EMOTION] = phase_feedback(PHASE_EMOTION, started_at + timedelta(minutes=2), answers)
    if RNG.random() < 0.78:
        answers = {
            "confidence_describing_visual_style": maybe_skip(scores["confidence_describing_visual_style"], 0.07),
            "familiarity_with_art_terms": maybe_skip(scores["familiarity_with_art_terms"], 0.09),
            "knows_how_to_describe_look": maybe_skip(scores["knows_how_to_describe_look"], 0.07),
        }
        stage_feedback[PHASE_ARTISTIC] = phase_feedback(PHASE_ARTISTIC, started_at + timedelta(minutes=7), answers)
    if RNG.random() < 0.72:
        answers = {
            "output_makes_uncomfortable": maybe_skip(scores["output_makes_uncomfortable"], 0.05),
            "satisfaction_with_output": maybe_skip(scores["satisfaction_with_output"], 0.06),
            "output_represents_people_fairly": maybe_skip(scores["output_represents_people_fairly"], 0.06),
            "needs_more_guidance_examples": maybe_skip(scores["needs_more_guidance_examples"], 0.08),
        }
        stage_feedback[PHASE_CODE] = phase_feedback(PHASE_CODE, started_at + timedelta(minutes=13), answers)
    return stage_feedback


def make_session(index: int, started_at: datetime) -> dict:
    profile = weighted_choice(
        [
            ("confident", 0.32),
            ("mixed", 0.26),
            ("low_tech", 0.18),
            ("high_tech_low_art", 0.14),
            ("emotion_uncertain", 0.10),
        ]
    )
    session_id = str(uuid.uuid4())
    survey = make_survey(started_at, profile)
    ai_experience = survey.get("ai_experience") if survey else None
    scores = make_scores(profile, ai_experience)
    turns = make_turns(started_at, scores, profile)
    stage_feedback = make_stage_feedback(started_at, scores)
    return {
        "session_id": session_id,
        "started_at": started_at.isoformat(),
        "turns": turns,
        "survey": survey,
        "stage_feedback": stage_feedback,
        "synthetic": {
            "profile": profile,
            "generator": "unified_project/scripts/generate_synthetic_logs.py",
            "index": index,
        },
    }


def main() -> None:
    if OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    last_started_at = datetime(2026, 5, 1, 12, 0, tzinfo=timezone.utc)
    start = last_started_at - timedelta(hours=(240 - 1) * 2)
    for index in range(240):
        started_at = start + timedelta(hours=index * 2, minutes=RNG.randint(0, 45))
        session = make_session(index, started_at)
        sid_prefix = session["session_id"][:8]
        timestamp = started_at.strftime("%Y%m%d_%H%M%S")
        path = OUT_DIR / f"session_{timestamp}_{sid_prefix}.json"
        path.write_text(json.dumps(session, indent=2), encoding="utf-8")

    readme = OUT_DIR / "README.md"
    readme.write_text(
        "# Synthetic logs\n\n"
        "Generated by `python scripts/generate_synthetic_logs.py` for dashboard testing. "
        "These files contain no real user data. The generated sessions end on 2026-05-01.\n\n"
        "The generated turns include synthetic `safety` objects that mirror the app guardrail schema: "
        "safe turns, flagged user-input or model-output signals, blocked user inputs, and held generated outputs.\n",
        encoding="utf-8",
    )
    print(f"Wrote 240 synthetic logs to {OUT_DIR}")


if __name__ == "__main__":
    main()
