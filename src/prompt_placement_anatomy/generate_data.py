"""Generate deterministic sample markdown files for the placement experiment.

Creates five markdown files in data/sample_files/. Each file has a known number
of TODO markers (case-insensitive "TODO" substring search):

    file_1.md: 2 TODOs
    file_2.md: 3 TODOs
    file_3.md: 1 TODO
    file_4.md: 0 TODOs
    file_5.md: 4 TODOs

This script is idempotent — running it multiple times produces the same output.
All scripts assume CWD is the project root (data/sample_files/ is relative to CWD).

Usage:
    python -m prompt_placement_anatomy.generate_data
"""

import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

OUTPUT_DIR = Path("data") / "sample_files"

# ---------------------------------------------------------------------------
# File content — 5 files, deterministic TODO counts
# ---------------------------------------------------------------------------

_FILES: list[tuple[str, str]] = [
    # -----------------------------------------------------------------------
    # file_1.md — 2 TODOs
    # -----------------------------------------------------------------------
    (
        "file_1.md",
        """\
# API Gateway — Project Notes

## Overview

This document tracks ongoing decisions for the API gateway migration project.
The gateway routes all traffic from legacy monolith endpoints to the new
microservices cluster. The migration is being done incrementally using a
strangler-fig pattern so that individual service owners can cut over without
a coordinated freeze.

## Current Status

The authentication middleware has been integrated and passes all unit tests.
Rate limiting is active on public endpoints using a token-bucket algorithm
with a default burst of 200 requests per second per client IP. Internal
service-to-service calls bypass rate limiting via a shared HMAC header.

## Open Items

- The circuit breaker threshold is currently set to 50 % failure rate over a
  10-second window. The payment team raised a concern that this is too sensitive
  for their batch settlement endpoint, which has naturally bursty error rates
  at month-end.
  <!-- TODO: review circuit breaker thresholds with the payment team -->

- The Prometheus metrics exporter was wired up for the legacy gateway but has
  not yet been connected to the new gateway binary. Until this is done, the
  SLO dashboards will show stale data and any alerts will be unreliable.
  <!-- TODO: wire up Prometheus exporter to the new gateway -->

## Decisions Made

- gRPC for internal service-to-service communication (protobuf v3 schemas).
- REST + JSON for external client-facing APIs (OpenAPI 3.1 spec in repo).
- 30-day log retention in object storage; 7-day hot retention in Elasticsearch.
- Canary deployments via weighted routing at 5 % / 25 % / 100 % stages.

## References

- Architecture Decision Record 0042: API Gateway Technology Selection
- Runbook: Gateway Deployment Checklist (internal wiki, page GW-42)
- Prometheus alerting rules — pull request #317
""",
    ),
    # -----------------------------------------------------------------------
    # file_2.md — 3 TODOs
    # -----------------------------------------------------------------------
    (
        "file_2.md",
        """\
# Weekly Engineering Meeting — Notes

**Date:** 2026-05-07
**Facilitator:** Priya S.
**Attendees:** Priya S., Tobias M., Anika R., Luca D., Yuki F.

## Agenda

1. Sprint review
2. Infrastructure incidents from the past week
3. Q2 roadmap check-in
4. On-call rotation update

## Sprint Review

The search re-ranking feature shipped to 10 % of users on Monday. Early metrics
look positive: click-through rate is up 4 % in the experiment bucket compared to
control. Full rollout is planned for Thursday if the error rate stays below 0.1 %.

The data-export pipeline was delayed due to a conflict in the database schema
migration. Two teams submitted migrations that modify the same table without
coordination, causing a merge conflict in the migration history.
<!-- TODO: resolve schema migration conflict blocking the data-export pipeline -->

## Infrastructure Incidents

A disk-space alert fired on the staging database at 02:14 UTC on Wednesday.
The on-call engineer cleared space by removing stale query-plan cache files.
Root cause is the autovacuum configuration, which has not been tuned since
the cluster was upgraded to Postgres 16 three months ago.
<!-- TODO: tune autovacuum settings for Postgres 16 on the staging cluster -->

Monitoring showed two brief latency spikes (p99 > 500 ms) on the image-resize
service. Both correlated with stop-the-world garbage collection pauses in the
JVM. A G1GC tuning session with the platform team is needed.
<!-- TODO: schedule G1GC tuning session for the image-resize service -->

## Roadmap Check-in

Q2 milestones are broadly on track. The mobile offline-mode feature carries the
highest risk; the team is blocked waiting for the client-side sync library to
reach API stability. Estimated slip is one sprint if the library is not ready
by end of next week.

## Action Items

- Priya: circulate updated capacity plan by end of week.
- Tobias: coordinate with mobile team on sync library timeline.
- Anika: draft runbook for autovacuum incident response.
- Luca: open ticket to track G1GC investigation.
""",
    ),
    # -----------------------------------------------------------------------
    # file_3.md — 1 TODO
    # -----------------------------------------------------------------------
    (
        "file_3.md",
        """\
# Feature Spec: User Notification Preferences

## Purpose

Allow users to control which notification types they receive and through which
delivery channel (email, push notification, or SMS). This spec covers the
backend storage model, the preference API, and the delivery-filter logic.

## Data Model

Preferences are stored per user in a dedicated `notification_preferences` table.

| Column       | Type          | Notes                              |
|---|---|---|
| user_id      | UUID          | FK to users table, indexed         |
| channel      | ENUM          | email, push, sms                   |
| event_type   | VARCHAR(64)   | e.g. order_shipped, comment_reply  |
| enabled      | BOOLEAN       | default true                       |
| updated_at   | TIMESTAMP     | auto-updated on write              |

A missing row is treated as "enabled" for backward compatibility with existing
users who were never prompted to set preferences.

## API

`GET /v1/me/notification-preferences` — returns the full preference list for
the authenticated user as a JSON array.

`PATCH /v1/me/notification-preferences` — upserts one or more preferences in
a single request. Both endpoints require a valid session token. Bulk updates
are capped at 50 entries per request to prevent abuse.

## Delivery Filter

Before dispatching any notification, the delivery service calls
`PreferenceService.is_enabled(user_id, channel, event_type)`. If the result
is `False`, the notification is silently dropped and a suppression counter is
incremented in the metrics store for observability.

## Known Gaps

<!-- TODO: define retention policy for preference-change audit logs -->

Audit logging is stubbed out in the current implementation. We need to decide
how long preference-change events are retained and whether they fall under the
same GDPR deletion rules as user-generated content.

## Acceptance Criteria

- Users can disable any event-type / channel combination independently.
- A disabled preference takes effect within 60 seconds of being saved.
- Preference changes appear in the audit log within 5 minutes.
- Suppression counters are visible in the Grafana notification dashboard.
""",
    ),
    # -----------------------------------------------------------------------
    # file_4.md — 0 TODOs
    # -----------------------------------------------------------------------
    (
        "file_4.md",
        """\
# System Architecture — Data Platform

## Overview

The data platform ingests events from production services, transforms them into
analytical models, and serves query results to internal dashboards and data
science workflows. It is designed for high throughput with eventual consistency
rather than low-latency transactional guarantees.

## Components

### Event Ingestion

Producers publish domain events to Kafka topics using Avro schemas registered
in the Schema Registry. Each topic is configured with a 7-day retention period
and a replication factor of three across availability zones.

### Stream Processing

Flink jobs consume from Kafka, apply stateful transformations (session
enrichment, deduplication, join with reference tables), and write output to
the data lake in Parquet format partitioned by event date and UTC hour.

### Data Lake

Object storage organised into three logical layers:

- **Raw** — byte-for-byte copies of ingested events, compressed with Snappy.
- **Curated** — cleaned, deduplicated, and enriched records in columnar format.
- **Aggregated** — pre-computed metrics and rollups for dashboarding.

### Query Layer

Apache Spark handles batch queries and exploratory workloads. Trino provides
interactive SQL access over the curated and aggregated layers. Frequently-run
dashboard queries are cached in Redis with a 5-minute TTL to reduce Trino load.

## Security

All data in transit is encrypted with TLS 1.3. Data at rest uses server-side
encryption with customer-managed keys stored in a dedicated key-management
service. Access is governed by attribute-based access control policies that are
reviewed and re-certified quarterly.

## Service Level Objectives

- Event ingestion lag: < 30 s (p99) under normal operating load.
- Query response time for pre-aggregated metrics: < 2 s (p95).
- Data lake availability: 99.9 % per calendar month.

## Diagram

The current architecture diagram is maintained in the internal wiki under
Data Platform — Architecture Overview and is reviewed at the start of each
quarter by the platform leads.
""",
    ),
    # -----------------------------------------------------------------------
    # file_5.md — 4 TODOs
    # -----------------------------------------------------------------------
    (
        "file_5.md",
        """\
# Sprint 34 — Planning Notes

**Sprint dates:** 2026-05-12 to 2026-05-23
**Team:** Platform Engineering
**Capacity:** 38 story points (two engineers on partial allocation)

## Sprint Goals

- Ship the background job-queue rewrite to production.
- Reduce p99 latency on the search API below 200 ms.
- Clear the four oldest items from the technical debt register.

## Backlog Items Pulled In

### Job Queue Rewrite

The existing cron-based job runner has reliability issues under high load:
jobs pile up silently when workers are saturated, and there is no retry
mechanism for transient failures. The rewrite uses Redis Streams for durable
at-least-once delivery with a dead-letter channel for permanently failed jobs.

<!-- TODO: write integration tests for dead-letter channel behaviour -->

Deployment checklist needs updating to cover the Redis Streams configuration
steps, including stream naming conventions and consumer group initialisation.

<!-- TODO: update deployment checklist with Redis Streams configuration steps -->

### Search Latency

Profiling identified two bottlenecks: a synchronous call to the recommendation
service on every search request (now converted to an async fire-and-forget
call) and a missing index on the `category_id` filter column.

<!-- TODO: add database index on search_items.category_id in migration 0089 -->

The async refactor is merged. The index migration script is drafted but has
not yet received a review from the database reliability team.

### Technical Debt

Four items from the debt register are scheduled this sprint:

1. Remove the deprecated `v1/legacy-export` endpoint (sunset date was Q1 2025).
2. Replace MD5 password hashing with bcrypt in the admin user module.
3. Consolidate duplicate logging utilities into a shared internal library.
4. Upgrade the internal HTTP client from requests 2.28 to httpx.

<!-- TODO: coordinate with the security team before removing the legacy-export endpoint -->

Item 2 must be coordinated with the security team because a credential rotation
is planned for May 19 and both changes touch the same authentication path.

## Risks

- The Redis Streams rollout depends on the infrastructure team provisioning
  a new cluster by May 14. Any slip pushes the job-queue work to sprint 35.
- The search index migration requires a maintenance window; scheduling is
  pending approval from the on-call rotation lead.

## Definition of Done

All items require passing CI, at least one peer review approval, and updated
runbook entries before being marked complete in the sprint board.
""",
    ),
]

# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

EXPECTED_COUNTS: list[int] = [2, 3, 1, 0, 4]


def generate() -> None:
    """Write all sample markdown files to OUTPUT_DIR.

    Creates the directory if it does not exist. Overwrites existing files
    so the script is idempotent.
    """
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for (filename, content), expected in zip(_FILES, EXPECTED_COUNTS):
        path = OUTPUT_DIR / filename
        path.write_text(content, encoding="utf-8")
        # Programmatic verification: count case-insensitive "TODO" occurrences.
        actual = content.upper().count("TODO")
        if actual != expected:
            logger.error(
                "BUG in generate_data: %s has %d TODOs, expected %d",
                filename,
                actual,
                expected,
            )
        else:
            logger.info("Wrote %s (%d TODOs)", path, actual)
    logger.info("Done. Generated %d files in %s", len(_FILES), OUTPUT_DIR)


if __name__ == "__main__":
    generate()
