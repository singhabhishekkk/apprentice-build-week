"""The drift panel, standalone and runnable.

This is the feature Apprentice shipped during OpenAI Build Week (2026-07-13 to 07-21),
written by driving Codex. The production version lives in a private monorepo behind auth,
multi-tenancy and a job queue. This file is the same two endpoints and the same rules,
over a seeded SQLite database, so a judge can run it in one command and see it work.

The rules it enforces are the ones that matter, and they are not cosmetic:

  * `drift` counts only rows whose source is a captured trace. Uploads and synthetic rows
    are not production traffic and must never appear on a traffic chart.

  * `retrain-candidate` counts GOLD rows only. Training reads gold only, so counting
    silver would overstate what a retrain would actually learn from. (This is the bug the
    human review caught after Codex's tests went green: the tests encoded the same wrong
    contract the brief did. See the README.)

  * The cutoff for "new" rows is when the last training job STARTED, not when it was
    queued: the worker snapshots rows at start, so anything created after that was never
    seen by the model.

  * `eligible` is false unless a retrain would actually clear the same gate the training
    endpoint enforces (MIN_TRAIN_ROWS). Offering a button that the API would reject with a
    400 is a dead end, and a dead end is a bug.

Run:  uv run uvicorn app:app --reload    (then open http://localhost:8000)
"""

from __future__ import annotations

from datetime import date, datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Mapped, Session, mapped_column

DB_PATH = Path(__file__).parent / "drift-demo.db"
ENGINE = create_engine(f"sqlite:///{DB_PATH}")

# The production gate. A task cannot train below this many gold rows, so the panel must
# not offer a retrain below it either.
MIN_TRAIN_ROWS = 500


class Base(DeclarativeBase):
    pass


class Task(Base):
    __tablename__ = "tasks"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    name: Mapped[str] = mapped_column(String(120), unique=True)


class Row(Base):
    """One captured input/output pair.

    `tier` is the trust level: gold means a human verified it, silver means it passed
    deterministic checks, raw means neither. `source` distinguishes production traffic
    (trace) from an upload.
    """

    __tablename__ = "rows"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    input: Mapped[str] = mapped_column(Text)
    output: Mapped[str] = mapped_column(Text)
    tier: Mapped[str] = mapped_column(String(10), default="raw", index=True)
    source: Mapped[str] = mapped_column(String(20), default="upload")
    feedback_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


class Job(Base):
    __tablename__ = "jobs"
    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    task_id: Mapped[str] = mapped_column(ForeignKey("tasks.id"), index=True)
    kind: Mapped[str] = mapped_column(String(20))
    status: Mapped[str] = mapped_column(String(20))
    started_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True))


app = FastAPI(title="Apprentice drift panel (Build Week demo)")


def _utc(value: datetime) -> datetime:
    """SQLite hands back naive datetimes. Treat them as UTC rather than local time.

    Getting this wrong shifts every row into the wrong day bucket, which is exactly the
    kind of quiet error a drift chart would show as real behaviour.
    """
    return value.replace(tzinfo=timezone.utc) if value.tzinfo is None else value.astimezone(timezone.utc)


def _iso(value: datetime | None) -> str | None:
    return _utc(value).isoformat() if value else None


def _task(session: Session, name: str) -> Task:
    task = session.query(Task).filter(Task.name == name).one_or_none()
    if task is None:
        raise HTTPException(status_code=404, detail=f"No task named {name!r}. Run `python seed.py`.")
    return task


@app.get("/v1/tasks/{name}/drift")
def task_drift(name: str, days: int = 30) -> dict[str, Any]:
    """Daily captured traffic and the feedback your app reported, over 1 to 90 days.

    Every day in the window is present, zero-filled. A quiet day is a real fact and must
    read as a quiet day, not as a gap in the line.
    """
    days = max(1, min(days, 90))
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=days - 1)
    start_at = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_at = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)

    series: dict[date, dict[str, Any]] = {
        start_date
        + timedelta(days=offset): {
            "date": (start_date + timedelta(days=offset)).isoformat(),
            "traces": 0,
            "feedback_count": 0,
            "mean_feedback_score": None,
        }
        for offset in range(days)
    }

    with Session(ENGINE) as session:
        task = _task(session, name)
        traces = (
            session.query(Row.created_at, Row.feedback_score)
            .filter(
                Row.task_id == task.id,
                Row.source == "trace",  # production traffic only
                Row.created_at >= start_at,
                Row.created_at < end_at,
            )
            .all()
        )

    scores: dict[date, list[float]] = {}
    for created_at, feedback_score in traces:
        day = _utc(created_at).date()
        if day not in series:
            continue
        series[day]["traces"] += 1
        if feedback_score is not None:
            series[day]["feedback_count"] += 1
            scores.setdefault(day, []).append(feedback_score)

    for day, values in scores.items():
        series[day]["mean_feedback_score"] = sum(values) / len(values)

    return {"task": name, "days": days, "series": list(series.values())}


@app.get("/v1/tasks/{name}/retrain-candidate")
def retrain_candidate(name: str) -> dict[str, Any]:
    """Gold rows the last training run never saw, and whether a retrain would pass its gate.

    Gold only. Training reads gold only, so silver rows would never reach the model and
    counting them here would overstate the payload on a panel whose entire job is to be
    trusted.
    """
    with Session(ENGINE) as session:
        task = _task(session, name)

        last_train = (
            session.query(Job)
            .filter(Job.task_id == task.id, Job.kind == "train", Job.status == "succeeded")
            .order_by(func.coalesce(Job.finished_at, Job.created_at).desc())
            .first()
        )

        gold = session.query(func.count(Row.id)).filter(Row.task_id == task.id, Row.tier == "gold")
        total_gold = gold.scalar() or 0

        new_gold = total_gold
        if last_train is not None:
            # The worker snapshots rows when the job STARTS, so rows created after that
            # start were never trained on. Queue time would be the wrong cutoff.
            snapshot_at = last_train.started_at or last_train.created_at
            new_gold = gold.filter(Row.created_at > snapshot_at).scalar() or 0

        return {
            "task": name,
            "last_trained_at": _iso(last_train.finished_at) if last_train else None,
            "new_gold_rows": new_gold,
            "total_gold_rows": total_gold,
            "min_train_rows": MIN_TRAIN_ROWS,
            # A retrain only helps if there is new gold data, and it only STARTS if the
            # task clears the same gate the training endpoint enforces.
            "eligible": new_gold > 0 and total_gold >= MIN_TRAIN_ROWS,
        }


@app.get("/")
def index() -> FileResponse:
    return FileResponse(Path(__file__).parent / "static" / "index.html")
