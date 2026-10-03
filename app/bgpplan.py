"""Confirmable BGP announcement plans pinned to a pool's current state.

The network team must not activate BGP announcements that were vetted
against an outdated view of the address pool: allocations come and go, and
a route that was correct when the plan was drawn up may be stale by the
time it is published.  A *plan* freezes the check in two steps.

``POST /bgp/announcement-plans``
    The caller submits a ``plan_id`` (idempotency key), the ``pool_id`` the
    announcements are checked against, the AS relationships and the
    announcement batch.  The existing stateless path audit is reused
    unchanged, then -- inside one ``BEGIN IMMEDIATE`` write transaction --
    the pool's exclusions and allocations are pinned to a consistent
    snapshot and the plan is stored only when every check passes:

    * every announced prefix lies completely inside the pool and does not
      overlap any exclusion (regardless of the announcement's audit
      status);
    * every current allocation is covered by at least one announcement
      that passed the audit (``status == "valid"``);
    * no allocation is covered by announcements from different origins.

    The stored plan is immutable: the audit results plus a digest of the
    pinned pool state are written once and never updated.

``POST /bgp/announcement-plans/{plan_id}/activate``
    Re-checks, inside a single write transaction, that the pool state
    digest still matches and the coverage relation still holds; only then
    is the plan marked ``active``.  Any change is a 409 conflict and the
    transaction rolls back, so nothing is partially applied.

``plan_id`` is the idempotency key: replaying with the same parameters
returns the original result; the same key with different parameters is a
409 conflict.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
import sqlite3

from .bgpaudit import BgpAuditError, audit
from .cidrmath import parse_network
from .db import write_transaction
from .service import (
    ApiError,
    _pool_allocations,
    _pool_exclusions,
    _row_pool,
    utc_now,
)
from .snapshot import canonical_bytes


def _plan_error(detail: str, code: str) -> ApiError:
    return ApiError(422, detail, code=code)


def _canonical_request(
    pool_id: int,
    relationships,
    parsed: list[tuple[str, ipaddress.IPv4Network, list[int]]],
) -> dict:
    """Canonical form of the plan parameters, for the idempotency hash.

    Relationship order and exact duplicate declarations are irrelevant (the
    audit treats them as a set), and announcement order is irrelevant as
    well (the audit sorts its results), so both are normalised before
    hashing.  The ``as_path`` is kept verbatim: a prepended path audits the
    same but is a different request.
    """

    relations = set()
    for rel in relationships:
        if rel.type == "customer_provider":
            relations.add(("customer_provider", rel.customer, rel.provider))
        else:
            low, high = sorted((rel.a, rel.b))
            relations.add(("peer", low, high))
    announcements = sorted(
        (
            {"id": ann_id, "prefix": str(net), "as_path": path}
            for ann_id, net, path in parsed
        ),
        key=lambda a: a["id"],
    )
    return {
        "pool_id": pool_id,
        "relationships": [list(rel) for rel in sorted(relations)],
        "announcements": announcements,
    }


def _pool_state_digest(
    pool: sqlite3.Row,
    exclusion_rows: list[sqlite3.Row],
    allocation_rows: list[sqlite3.Row],
) -> str:
    """SHA-256 over the canonical pool state the plan is pinned to.

    Covers the pool CIDR, the exclusion CIDRs and the allocations (CIDR,
    request_id, prefix length), all ordered by numeric network address so
    the digest does not depend on insertion order.
    """

    data = {
        "cidr": pool["cidr"],
        "exclusions": sorted(
            (row["cidr"] for row in exclusion_rows),
            key=lambda c: int(parse_network(c).network_address),
        ),
        "allocations": sorted(
            (
                {
                    "cidr": row["cidr"],
                    "request_id": row["request_id"],
                    "prefix_length": row["prefix_length"],
                }
                for row in allocation_rows
            ),
            key=lambda a: int(parse_network(a["cidr"]).network_address),
        ),
    }
    return hashlib.sha256(canonical_bytes(data)).hexdigest()


def _valid_announcements(results: list[dict]) -> list[dict]:
    """Audit-passing announcements, as coverage candidates."""

    return [
        {
            "id": r["id"],
            "network": parse_network(r["prefix"]),
            "origin": r["origin"],
        }
        for r in results
        if r["status"] == "valid"
    ]


def _coverage_error(
    allocations: list[tuple[ipaddress.IPv4Network, str]],
    valid: list[dict],
) -> tuple[str, str] | None:
    """First coverage violation, or ``None`` when the coverage holds.

    Every current allocation must be covered by at least one valid
    announcement, and all valid announcements covering one allocation must
    agree on the origin AS.
    """

    for net, request_id in allocations:
        covering = [a for a in valid if net.subnet_of(a["network"])]
        if not covering:
            return (
                f"allocation {net} (request_id {request_id!r}) is not covered "
                "by any valid announcement",
                "plan_allocation_uncovered",
            )
        origins = {a["origin"] for a in covering}
        if len(origins) > 1:
            return (
                f"allocation {net} (request_id {request_id!r}) is covered by "
                f"announcements from different origins {sorted(origins)}",
                "plan_origin_conflict",
            )
    return None


def _replay_result(row: sqlite3.Row) -> dict:
    return {
        "plan_id": row["plan_id"],
        "pool_id": row["pool_id"],
        "status": row["status"],
        "pool_digest": row["pool_digest"],
        "results": json.loads(row["plan"])["results"],
        "created_at": row["created_at"],
        "replayed": True,
    }


def create_plan(plan_id: str, pool_id: int, relationships, announcements) -> dict:
    """Validate and store an immutable announcement plan.

    The path audit is pure computation and runs before any transaction, so
    a rejected batch never opens one.  The pool snapshot, the pool-dependent
    checks and the insert all run inside one ``BEGIN IMMEDIATE`` write
    transaction, so the pinned digest always matches the state the checks
    ran against -- a plan can never be stored against a pool state that
    changed underneath it.
    """

    # 1. Reuse the stateless path audit (pure computation, no I/O).
    try:
        audited = audit(relationships, announcements)
    except BgpAuditError as exc:
        raise ApiError(422, exc.detail, code=exc.code)

    # 2. Parse the announced prefixes (the audit already validated them)
    #    and fix the canonical request for the idempotency hash.
    parsed = [
        (ann.id, parse_network(ann.prefix), list(ann.as_path))
        for ann in announcements
    ]
    request_hash = hashlib.sha256(
        canonical_bytes(_canonical_request(pool_id, relationships, parsed))
    ).hexdigest()
    valid = _valid_announcements(audited["results"])

    # 3. Pin the pool state, run the pool-dependent checks and store the
    #    plan -- atomically.
    with write_transaction() as conn:
        previous = conn.execute(
            "SELECT * FROM announcement_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if previous is not None:
            if previous["request_hash"] != request_hash:
                raise ApiError(
                    409,
                    f"plan_id {plan_id!r} was already used for a different "
                    "announcement plan",
                    code="plan_id_conflict",
                )
            return _replay_result(previous)

        pool = _row_pool(conn, pool_id)
        pool_net = parse_network(pool["cidr"])
        exclusion_rows = _pool_exclusions(conn, pool_id)
        allocation_rows = _pool_allocations(conn, pool_id)
        exclusion_nets = [parse_network(r["cidr"]) for r in exclusion_rows]

        # Every announced prefix must lie completely inside the pool and
        # must not overlap any exclusion -- regardless of its audit status.
        for ann_id, net, _path in parsed:
            if not net.subnet_of(pool_net):
                raise _plan_error(
                    f"announcement {ann_id!r} prefix {net} is not inside "
                    f"pool {pool_net}",
                    "plan_prefix_outside_pool",
                )
            for exclusion in exclusion_nets:
                if net.overlaps(exclusion):
                    raise _plan_error(
                        f"announcement {ann_id!r} prefix {net} overlaps "
                        f"exclusion {exclusion}",
                        "plan_prefix_overlap_exclusion",
                    )

        allocations = [
            (parse_network(r["cidr"]), r["request_id"]) for r in allocation_rows
        ]
        error = _coverage_error(allocations, valid)
        if error is not None:
            raise _plan_error(*error)

        digest = _pool_state_digest(pool, exclusion_rows, allocation_rows)
        now = utc_now()
        conn.execute(
            "INSERT INTO announcement_plans "
            "(plan_id, pool_id, status, request_hash, pool_digest, plan, created_at) "
            "VALUES (?, ?, 'pending', ?, ?, ?, ?)",
            (
                plan_id,
                pool_id,
                request_hash,
                digest,
                json.dumps({"results": audited["results"]}),
                now,
            ),
        )

    return {
        "plan_id": plan_id,
        "pool_id": pool_id,
        "status": "pending",
        "pool_digest": digest,
        "results": audited["results"],
        "created_at": now,
        "replayed": False,
    }


def _activate_replay(row: sqlite3.Row) -> dict:
    return {
        "plan_id": row["plan_id"],
        "pool_id": row["pool_id"],
        "status": "active",
        "pool_digest": row["pool_digest"],
        "activated_at": row["activated_at"],
        "replayed": True,
    }


def activate_plan(plan_id: str) -> dict:
    """Mark a pending plan active when the pool state is unchanged.

    Everything runs inside one ``BEGIN IMMEDIATE`` write transaction: the
    pool state digest and the coverage relation are re-checked against the
    current allocations, and only when both still hold is the plan marked
    ``active``.  Any change is a 409 conflict; the transaction rolls back,
    so nothing is partially applied.  Activating an already-active plan
    replays the original result.
    """

    with write_transaction() as conn:
        row = conn.execute(
            "SELECT * FROM announcement_plans WHERE plan_id = ?",
            (plan_id,),
        ).fetchone()
        if row is None:
            raise ApiError(
                404,
                f"announcement plan {plan_id!r} not found",
                code="plan_not_found",
            )
        if row["status"] == "active":
            return _activate_replay(row)

        pool = _row_pool(conn, row["pool_id"])
        exclusion_rows = _pool_exclusions(conn, row["pool_id"])
        allocation_rows = _pool_allocations(conn, row["pool_id"])

        current = _pool_state_digest(pool, exclusion_rows, allocation_rows)
        if current != row["pool_digest"]:
            raise ApiError(
                409,
                f"pool {row['pool_id']} changed since plan {plan_id!r} was "
                "created; create a new plan for the current state",
                code="plan_stale",
            )

        # The digest pins the allocations, so coverage necessarily still
        # holds; re-check it explicitly so the guarantee does not depend on
        # the digest construction alone.
        valid = _valid_announcements(json.loads(row["plan"])["results"])
        allocations = [
            (parse_network(r["cidr"]), r["request_id"]) for r in allocation_rows
        ]
        error = _coverage_error(allocations, valid)
        if error is not None:
            raise ApiError(
                409,
                f"plan {plan_id!r} no longer covers the current allocations: "
                f"{error[0]}",
                code="plan_stale",
            )

        now = utc_now()
        conn.execute(
            "UPDATE announcement_plans SET status = 'active', activated_at = ? "
            "WHERE plan_id = ?",
            (now, plan_id),
        )

    return {
        "plan_id": plan_id,
        "pool_id": row["pool_id"],
        "status": "active",
        "pool_digest": row["pool_digest"],
        "activated_at": now,
        "replayed": False,
    }
