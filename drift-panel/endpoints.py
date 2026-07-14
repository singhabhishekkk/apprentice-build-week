"""The drift panel endpoints, verbatim from the production monorepo.

Copied from `apprentice-api/src/apprentice_api/main.py` so that the code being judged is
the code that actually ships, not a reimplementation. It will not run on its own: it needs
the surrounding app (auth, the DB session, the collaborator ACL). For something runnable,
see ../drift-demo.

Written by driving Codex (session 019f5eb6-27b4-7e00-af1b-04285e89a907), then corrected in
human review. The three defects that review caught, which the green tests did not, are
described in the README.
"""

@app.get("/v1/tasks/{task_id}/drift")
def task_drift(
    task_id: str,
    days: int = 30,
    user: AuthUser = Depends(get_current_user),
) -> dict[str, Any]:
    from apprentice_api import collaborators

    days = max(1, min(days, 90))
    today = datetime.now(timezone.utc).date()
    start_date = today - timedelta(days=days - 1)
    start_at = datetime.combine(start_date, datetime.min.time(), tzinfo=timezone.utc)
    end_at = datetime.combine(today + timedelta(days=1), datetime.min.time(), tzinfo=timezone.utc)
    series: dict[date, dict[str, Any]] = {
        start_date + timedelta(days=offset): {
            "date": (start_date + timedelta(days=offset)).isoformat(),
            "traces": 0,
            "feedback_count": 0,
            "mean_feedback_score": None,
        }
        for offset in range(days)
    }

    with db.get_session() as session:
        collaborators.authorize_task_by_id(session, user, task_id, "read")
        traces = (
            session.query(db.Row.created_at, db.Row.feedback_score)
            .filter(
                db.Row.task_id == task_id,
                db.Row.source == "trace",
                db.Row.created_at >= start_at,
                db.Row.created_at < end_at,
            )
            .all()
        )

    feedback_scores: dict[date, list[float]] = {}
    for created_at, feedback_score in traces:
        created_at_utc = (
            created_at.replace(tzinfo=timezone.utc)
            if created_at.tzinfo is None
            else created_at.astimezone(timezone.utc)
        )
        day = created_at_utc.date()
        series[day]["traces"] += 1
        if feedback_score is not None:
            series[day]["feedback_count"] += 1
            feedback_scores.setdefault(day, []).append(feedback_score)
    for day, scores in feedback_scores.items():
        series[day]["mean_feedback_score"] = sum(scores) / len(scores)

    return {"task_id": task_id, "days": days, "series": list(series.values())}


@app.get("/v1/tasks/{task_id}/retrain-candidate")
def retrain_candidate(
    task_id: str, user: AuthUser = Depends(get_current_user)
) -> dict[str, Any]:
    """Gold rows added since the last successful train, and whether a retrain would pass its gate.

    Gold only: both the train gate and the train worker select `tier == "gold"` rows,
    so silver rows would never reach the model and must not be counted here.
    """
    from apprentice_api import collaborators

    with db.get_session() as session:
        collaborators.authorize_task_by_id(session, user, task_id, "read")
        last_train = (
            session.query(db.Job)
            .filter(
                db.Job.task_id == task_id,
                db.Job.kind == "train",
                db.Job.status == "succeeded",
            )
            .order_by(func.coalesce(db.Job.finished_at, db.Job.created_at).desc())
            .first()
        )
        gold = session.query(func.count(db.Row.id)).filter(
            db.Row.task_id == task_id,
            db.Row.tier == "gold",
        )
        total_gold = gold.scalar() or 0
        new_gold = total_gold
        if last_train is not None:
            # The worker snapshots rows when the job starts, so rows created after that
            # start were never trained on. ponytail: `created_at` still misses rows
            # re-tiered to gold after the run; add a `verified_at` column to close that.
            snapshot_at = last_train.started_at or last_train.created_at
            new_gold = gold.filter(db.Row.created_at > snapshot_at).scalar() or 0

    return {
        "task_id": task_id,
        "last_train_job_id": last_train.id if last_train else None,
        "last_trained_at": _iso(last_train.finished_at or last_train.created_at)
        if last_train
        else None,
        "new_gold_rows": new_gold,
        "total_gold_rows": total_gold,
        "min_train_rows": MIN_TRAIN_ROWS,
        # A retrain only helps if there is new gold data, and it only starts if the task
        # clears the same gate POST /v1/tasks/{task}/train enforces.
        "eligible": new_gold > 0 and total_gold >= MIN_TRAIN_ROWS,
    }
