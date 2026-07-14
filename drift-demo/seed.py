"""Seeds a task whose quality is visibly decaying, which is the thing a drift panel is for.

30 days of captured traffic on a support-triage task. The model was trained on day 18 and
was good; since then the inputs have drifted and the feedback your app reports has fallen
from ~0.95 to ~0.55. Enough gold rows have accumulated since that training run that a
retrain is now worth doing, and the panel says so.

The numbers are generated, not measured. They are here so the endpoints have something
real to chart; nothing in this file is presented as a benchmark result. The measured
numbers in this repo are the ones in benchmark/, which you can reproduce yourself.

Run:  uv run python seed.py
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from uuid import uuid4

from app import DB_PATH, ENGINE, Base, Job, Row, Task
from sqlalchemy.orm import Session

TASK = "support-triage"
DAYS = 30
TRAINED_DAYS_AGO = 12

rng = random.Random(42)  # deterministic: two people running this see the same chart


def _id() -> str:
    return uuid4().hex[:32]


def main() -> None:
    if DB_PATH.exists():
        DB_PATH.unlink()
    Base.metadata.create_all(ENGINE)

    now = datetime.now(timezone.utc)
    task = Task(id=_id(), name=TASK)

    rows: list[Row] = []
    for day in range(DAYS):
        day_start = (now - timedelta(days=DAYS - 1 - day)).replace(hour=0, minute=0, second=0, microsecond=0)

        # Traffic grows slowly, with weekends quieter. A flat line would look synthetic.
        weekday = day_start.weekday()
        base = 28 + day
        traffic = int(base * (0.45 if weekday >= 5 else 1.0) * rng.uniform(0.85, 1.15))

        # The story: quality holds, then drifts down after the model was trained.
        progress = day / (DAYS - 1)
        quality = 0.95 - 0.40 * max(0.0, (progress - 0.35)) / 0.65

        for _ in range(traffic):
            created = day_start + timedelta(
                hours=rng.randint(7, 21), minutes=rng.randint(0, 59), seconds=rng.randint(0, 59)
            )
            # Only about half of production traffic gets rated by the app. Pretending every
            # call comes back with a score would be a lie about how integrations behave.
            rated = rng.random() < 0.45
            score = None
            if rated:
                score = max(0.0, min(1.0, rng.gauss(quality, 0.12)))

            # A slice of traffic gets verified by a human and becomes gold.
            tier = "gold" if rng.random() < 0.22 else "raw"

            rows.append(
                Row(
                    id=_id(),
                    task_id=task.id,
                    input="Customer email about a delayed refund on order #%d" % rng.randint(1000, 9999),
                    output='{"intent": "refund_status", "priority": "normal"}',
                    tier=tier,
                    source="trace",
                    feedback_score=score,
                    created_at=created,
                )
            )

    # Enough historical gold to clear MIN_TRAIN_ROWS, so the panel has a real decision to
    # make rather than being blocked on data volume.
    for _ in range(520):
        rows.append(
            Row(
                id=_id(),
                task_id=task.id,
                input="Historical verified example",
                output='{"intent": "refund_status", "priority": "normal"}',
                tier="gold",
                source="upload",
                feedback_score=None,
                created_at=now - timedelta(days=DAYS + rng.randint(1, 40)),
            )
        )

    started = now - timedelta(days=TRAINED_DAYS_AGO)
    train = Job(
        id=_id(),
        task_id=task.id,
        kind="train",
        status="succeeded",
        started_at=started,
        finished_at=started + timedelta(minutes=41),
        created_at=started - timedelta(minutes=3),
    )

    # Count BEFORE the commit: SQLAlchemy expires instances on commit, so reading these
    # attributes afterwards detaches and raises.
    traces = sum(1 for r in rows if r.source == "trace")
    rated = sum(1 for r in rows if r.feedback_score is not None)
    gold = sum(1 for r in rows if r.tier == "gold")
    new_gold = sum(1 for r in rows if r.tier == "gold" and r.created_at > started)

    with Session(ENGINE) as session:
        session.add(task)
        session.add(train)
        session.add_all(rows)
        session.commit()

    print(f"task           {TASK}")
    print(f"traces (30d)   {traces}")
    print(f"rated by app   {rated}")
    print(f"gold rows      {gold}  ({new_gold} since the last training run started)")
    print(f"last trained   {TRAINED_DAYS_AGO} days ago")
    print("\nuv run uvicorn app:app  ->  http://localhost:8000")


if __name__ == "__main__":
    main()
