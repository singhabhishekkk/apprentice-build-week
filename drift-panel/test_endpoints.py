"""The tests for those endpoints, verbatim from the production monorepo.

Copied from `apprentice-api/tests/test_api.py`. They rely on the app's fixtures, so they do
not run standalone; they are here to be read.

Worth reading the retrain-candidate tests specifically. An earlier version of them passed
against WRONG code, because they encoded the same wrong contract the brief handed to Codex
(gold + silver, when training reads gold only). Green tests prove the code runs. They do
not prove it is right.
"""


def test_drift_panel_zero_data_task(client: TestClient) -> None:
    task_id = client.post("/v1/tasks", json={"name": "empty"}).json()["id"]

    drift = client.get(f"/v1/tasks/{task_id}/drift").json()
    candidate = client.get(f"/v1/tasks/{task_id}/retrain-candidate").json()

    assert drift["days"] == 30
    assert len(drift["series"]) == 30
    assert all(
        day["traces"] == 0
        and day["feedback_count"] == 0
        and day["mean_feedback_score"] is None
        for day in drift["series"]
    )
    assert candidate == {
        "task_id": task_id,
        "last_train_job_id": None,
        "last_trained_at": None,
        "new_gold_rows": 0,
        "total_gold_rows": 0,
        "min_train_rows": 2,
        "eligible": False,
    }


def test_drift_panel_routes_hide_non_owner_task(client: TestClient) -> None:
    task_id = client.post("/v1/tasks", json={"name": "private-drift"}).json()["id"]
    _as_user("user-test-2", "b@example.com")

    assert client.get(f"/v1/tasks/{task_id}/drift").status_code == 404
    assert client.get(f"/v1/tasks/{task_id}/retrain-candidate").status_code == 404


def test_retrain_candidate_without_succeeded_train_counts_all_gold(
    client: TestClient,
) -> None:
    from apprentice_api import db

    task_id = client.post("/v1/tasks", json={"name": "never-trained"}).json()["id"]
    with db.get_session() as session:
        session.add_all(
            [
                db.Row(task_id=task_id, input="g1", output="o", tier="gold"),
                db.Row(task_id=task_id, input="g2", output="o", tier="gold"),
                db.Row(task_id=task_id, input="s", output="o", tier="silver"),
                db.Row(task_id=task_id, input="r", output="o", tier="raw"),
                db.Job(task_id=task_id, kind="train", status="failed"),
            ]
        )
        session.commit()

    body = client.get(f"/v1/tasks/{task_id}/retrain-candidate").json()

    # Training reads gold only, so silver and raw never count towards a retrain.
    assert body["last_train_job_id"] is None
    assert body["new_gold_rows"] == 2
    assert body["total_gold_rows"] == 2
    assert body["eligible"] is True


def test_retrain_candidate_not_eligible_below_min_train_rows(client: TestClient) -> None:
    from apprentice_api import db

    task_id = client.post("/v1/tasks", json={"name": "too-thin"}).json()["id"]
    with db.get_session() as session:
        session.add(db.Row(task_id=task_id, input="g", output="o", tier="gold"))
        session.commit()

    body = client.get(f"/v1/tasks/{task_id}/retrain-candidate").json()

    # New gold data exists, but POST /train would reject it, so the panel must not offer a retrain.
    assert body["new_gold_rows"] == 1
    assert body["total_gold_rows"] == 1
    assert body["min_train_rows"] == 2
    assert body["eligible"] is False


def test_retrain_candidate_counts_gold_rows_created_after_latest_train_started(
    client: TestClient,
) -> None:
    from apprentice_api import db

    task_id = client.post("/v1/tasks", json={"name": "trained"}).json()["id"]
    now = datetime.now(timezone.utc)
    latest_started = now - timedelta(days=2)
    latest_finished = now - timedelta(days=2) + timedelta(hours=1)
    with db.get_session() as session:
        older_train = db.Job(
            task_id=task_id,
            kind="train",
            status="succeeded",
            created_at=now - timedelta(days=5),
            started_at=now - timedelta(days=5),
            finished_at=now - timedelta(days=4),
        )
        latest_train = db.Job(
            task_id=task_id,
            kind="train",
            status="succeeded",
            created_at=now - timedelta(days=3),
            started_at=latest_started,
            finished_at=latest_finished,
        )
        session.add_all(
            [
                older_train,
                latest_train,
                # Queued before the run started, so it was in the training snapshot.
                db.Row(
                    task_id=task_id,
                    input="trained-gold",
                    output="o",
                    tier="gold",
                    created_at=now - timedelta(days=4),
                ),
                db.Row(
                    task_id=task_id,
                    input="new-gold-1",
                    output="o",
                    tier="gold",
                    created_at=now - timedelta(days=1),
                ),
                db.Row(
                    task_id=task_id,
                    input="new-gold-2",
                    output="o",
                    tier="gold",
                    created_at=now,
                ),
                db.Row(
                    task_id=task_id,
                    input="new-silver",
                    output="o",
                    tier="silver",
                    created_at=now,
                ),
            ]
        )
        session.commit()
        latest_train_id = latest_train.id

    body = client.get(f"/v1/tasks/{task_id}/retrain-candidate").json()

    assert body["last_train_job_id"] == latest_train_id
    assert body["last_trained_at"].startswith(latest_finished.replace(tzinfo=None).isoformat())
    # Cutoff is when the run started (the worker snapshots rows then), not when it was queued.
    assert body["new_gold_rows"] == 2
    assert body["total_gold_rows"] == 3
    assert body["eligible"] is True


# -- auth ----------------------------------------------------------------------
