"""API integration tests for confirmable BGP announcement plans."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

# Relationships reused across tests:
#   10 is a customer of 100; 100 peers 200.
#   [10, 100] -> origin 100 (valid); [200] / [100, 200] -> origin 200 (valid).
RELATIONSHIPS = [
    {"type": "customer_provider", "customer": 10, "provider": 100},
    {"type": "peer", "a": 100, "b": 200},
]


def _create_pool(client, cidr="10.0.0.0/24", exclusions=None):
    resp = client.post("/pools", json={"cidr": cidr, "exclusions": exclusions or []})
    assert resp.status_code == 201, resp.text
    return resp.json()["id"]


def _allocate(client, pool_id, request_id, prefix_length):
    resp = client.post(
        f"/pools/{pool_id}/allocations",
        json={"request_id": request_id, "prefix_length": prefix_length},
    )
    assert resp.status_code == 201, resp.text
    return resp.json()["cidr"]


def _create_plan(client, body):
    return client.post("/bgp/announcement-plans", json=body)


def _plan_body(pool_id, announcements, plan_id="plan-1", relationships=None):
    return {
        "plan_id": plan_id,
        "pool_id": pool_id,
        "relationships": RELATIONSHIPS if relationships is None else relationships,
        "announcements": announcements,
    }


def _activate(client, plan_id="plan-1"):
    return client.post(f"/bgp/announcement-plans/{plan_id}/activate")


# ---------------------------------------------------------------------------
# Plan creation
# ---------------------------------------------------------------------------


def test_create_plan_success(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)  # 10.0.0.0/25
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["plan_id"] == "plan-1"
    assert body["pool_id"] == pid
    assert body["status"] == "pending"
    assert len(body["pool_digest"]) == 64
    assert body["replayed"] is False
    assert [r["id"] for r in body["results"]] == ["a1"]
    assert body["results"][0]["status"] == "valid"
    assert body["results"][0]["origin"] == 100
    assert body["results"][0]["as_path"] == [10, 100]


def test_create_plan_pool_not_found(client):
    resp = _create_plan(
        client,
        _plan_body(999, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "pool_not_found"


def test_create_plan_prefix_outside_pool(client):
    pid = _create_pool(client)
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.1.0.0/24", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "plan_prefix_outside_pool"


def test_create_plan_prefix_overlaps_exclusion(client):
    pid = _create_pool(client, exclusions=["10.0.0.128/25"])
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "plan_prefix_overlap_exclusion"


def test_create_plan_allocation_uncovered(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)  # 10.0.0.0/25
    _allocate(client, pid, "req-b", 25)  # 10.0.0.128/25
    # The announcement covers only the first allocation.
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/25", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "plan_allocation_uncovered"


def test_create_plan_origin_conflict(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)  # 10.0.0.0/25
    # Both announcements are valid and cover the allocation, but their
    # origins differ.
    resp = _create_plan(
        client,
        _plan_body(
            pid,
            [
                {"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]},
                {"id": "a2", "prefix": "10.0.0.0/25", "as_path": [200]},
            ],
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "plan_origin_conflict"


def test_create_plan_same_origin_covering_is_accepted(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)  # 10.0.0.0/25
    # Two valid announcements cover the allocation, but both originate from
    # AS 100, so there is no origin conflict.
    resp = _create_plan(
        client,
        _plan_body(
            pid,
            [
                {"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]},
                {"id": "a2", "prefix": "10.0.0.0/25", "as_path": [100]},
            ],
        ),
    )
    assert resp.status_code == 201, resp.text


def test_create_plan_failed_audit_does_not_cover(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    # The only covering announcement is an AS loop, so it cannot cover.
    resp = _create_plan(
        client,
        _plan_body(
            pid,
            [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100, 10]}],
        ),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "plan_allocation_uncovered"


def test_create_plan_audit_error_passthrough(client):
    pid = _create_pool(client)
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 999]}]),
    )
    assert resp.status_code == 422
    assert resp.json()["error"]["code"] == "as_relation_unknown"


def test_create_plan_on_empty_pool(client):
    # No allocations: coverage is vacuously satisfied.
    pid = _create_pool(client)
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["status"] == "pending"


def test_create_plan_does_not_touch_pool(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    before = client.get(f"/pools/{pid}").json()
    resp = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    )
    assert resp.status_code == 201, resp.text
    assert client.get(f"/pools/{pid}").json() == before


# ---------------------------------------------------------------------------
# Idempotency
# ---------------------------------------------------------------------------


def test_create_plan_idempotent_replay(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    body = _plan_body(
        pid,
        [
            {"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]},
            {"id": "a2", "prefix": "10.0.0.128/25", "as_path": [200]},
        ],
    )
    first = _create_plan(client, body)
    assert first.status_code == 201, first.text

    replay = _create_plan(client, body)
    assert replay.status_code == 201
    assert replay.json()["replayed"] is True
    assert replay.json()["pool_digest"] == first.json()["pool_digest"]
    assert replay.json()["created_at"] == first.json()["created_at"]
    assert replay.json()["results"] == first.json()["results"]

    # Announcement order is part of the wire format, not the plan identity.
    shuffled = dict(body, announcements=list(reversed(body["announcements"])))
    again = _create_plan(client, shuffled)
    assert again.status_code == 201
    assert again.json()["replayed"] is True


def test_create_plan_id_conflict(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    body = _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}])
    assert _create_plan(client, body).status_code == 201

    # Same plan_id, different announcement.
    changed = _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [200]}])
    resp = _create_plan(client, changed)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "plan_id_conflict"

    # Same plan_id, different pool: the idempotency check wins over the
    # pool-existence check.
    other = _plan_body(999, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}])
    resp = _create_plan(client, other)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "plan_id_conflict"


def test_concurrent_create_same_plan_id_stores_once(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    body = _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}])

    from app.main import app

    barrier = threading.Barrier(8)
    responses = []
    lock = threading.Lock()

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = local.post("/bgp/announcement-plans", json=body)
        with lock:
            responses.append(resp)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r.status_code == 201 for r in responses)
    assert sum(1 for r in responses if not r.json()["replayed"]) == 1
    assert len({r.json()["pool_digest"] for r in responses}) == 1


# ---------------------------------------------------------------------------
# Activation
# ---------------------------------------------------------------------------


def test_activate_success_and_replay(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    created = _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    )
    assert created.status_code == 201

    resp = _activate(client)
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["plan_id"] == "plan-1"
    assert body["pool_id"] == pid
    assert body["status"] == "active"
    assert body["replayed"] is False
    assert body["pool_digest"] == created.json()["pool_digest"]
    assert body["activated_at"]

    again = _activate(client)
    assert again.status_code == 200
    assert again.json()["replayed"] is True
    assert again.json()["activated_at"] == body["activated_at"]


def test_activate_not_found(client):
    resp = _activate(client, "nope")
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "plan_not_found"


def test_activate_conflicts_after_new_allocation(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    assert _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    ).status_code == 201

    _allocate(client, pid, "req-b", 25)  # pool state changed
    resp = _activate(client)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "plan_stale"


def test_activate_conflicts_after_release(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    _allocate(client, pid, "req-b", 25)
    assert _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    ).status_code == 201

    client.delete(f"/pools/{pid}/allocations/req-b")
    resp = _activate(client)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "plan_stale"


def test_activate_conflicts_after_new_exclusion(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    assert _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    ).status_code == 201

    resp = client.post(f"/pools/{pid}/exclusions", json={"cidr": "10.0.0.192/26"})
    assert resp.status_code == 201
    resp = _activate(client)
    assert resp.status_code == 409
    assert resp.json()["error"]["code"] == "plan_stale"


def test_failed_activation_applies_nothing_and_recovers(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    assert _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    ).status_code == 201

    _allocate(client, pid, "req-b", 25)
    assert _activate(client).status_code == 409

    # The failed activation left the plan pending: once the pool returns to
    # the pinned state, the same plan still activates.
    client.delete(f"/pools/{pid}/allocations/req-b")
    resp = _activate(client)
    assert resp.status_code == 200, resp.text
    assert resp.json()["status"] == "active"
    assert resp.json()["replayed"] is False


def test_plan_dropped_with_pool_partition(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)  # 10.0.0.0/25, inside the first sub-pool
    assert _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    ).status_code == 201

    resp = client.post(
        f"/pools/{pid}/partitions",
        json={"operation_id": "split-1", "subnets": ["10.0.0.0/25", "10.0.0.128/25"]},
    )
    assert resp.status_code == 201, resp.text

    resp = _activate(client)
    assert resp.status_code == 404
    assert resp.json()["error"]["code"] == "plan_not_found"


def test_concurrent_activation_is_atomic(client):
    pid = _create_pool(client)
    _allocate(client, pid, "req-a", 25)
    assert _create_plan(
        client,
        _plan_body(pid, [{"id": "a1", "prefix": "10.0.0.0/24", "as_path": [10, 100]}]),
    ).status_code == 201

    from app.main import app

    barrier = threading.Barrier(8)
    results = []
    lock = threading.Lock()

    def worker():
        local = TestClient(app)
        barrier.wait()
        resp = local.post("/bgp/announcement-plans/plan-1/activate")
        with lock:
            results.append(resp.json())

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert all(r["status"] == "active" for r in results)
    assert sum(1 for r in results if not r["replayed"]) == 1
    assert len({r["activated_at"] for r in results}) == 1
