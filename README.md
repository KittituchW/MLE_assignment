# Netflix Discovery EDA

An EDA-aware recommendation reference system for the supplied 32,000-title metadata dataset. It turns real behavior
events into recency-weighted content profiles and returns personalized movies/TV. When a user has no history, it falls
back to a confidence-adjusted discovery score.

> The source data contains no users or viewing behavior. This repository does **not** manufacture behavior and claim
> that recovering it proves model quality. Supply events using the documented contract. The bundled 300-row catalog is
> only a smoke-test fixture.

## How it works

The preparation job applies findings from the supplied EDA:

- namespaces movie/TV IDs and deterministically collapses repeated TV IDs;
- treats zero ratings as missing and shrinks sparse ratings toward a type-specific prior;
- ranks popularity within movie/TV because their raw scales differ;
- excludes `duration`, redundant `rating`, and questionable `date_added` from modeling;
- uses genres, language, country, title, and description as sparse content features.

Behavior events (`impression`, `click`, `play`, `complete`, `like`, `dismiss`) have explicit weights and a 180-day
half-life. Weighted watched-item vectors form a user profile. Ranking is 85% cosine affinity and 15% robust discovery
score. Unknown users use the discovery score alone. See `docs/architecture.md` for the production design.

## Run

```bash
python -m venv .venv
. .venv/bin/activate                         # Windows: .venv\Scripts\activate
pip install -e '.[dev]'

discovery-reco build --catalog data/sample --output artifacts/model
RECO_ARTIFACT=artifacts/model uvicorn discovery_reco.api:app --host 0.0.0.0 --port 8000
pytest --cov=discovery_reco
```

For personalized results, pass `--events /path/events.csv`. Required columns are:

```csv
user_id,item_id,event_type,timestamp
A,movie_18823,complete,2026-01-01T10:00:00Z
A,movie_10195,like,2026-01-02T10:00:00Z
```

Evaluation holds out each eligible user's latest positive event:

```bash
discovery-reco evaluate --catalog /path/DSAssignmentDataSet --events /path/events.csv
```

## API request and response

```bash
curl -X POST http://localhost:8000/v1/recommendations \
  -H 'content-type: application/json' \
  -d '{"user_id":"A","limit":3,"content_type":"movie"}'
```

```json
{
  "user_id": "A",
  "strategy": "behavior_content_profile",
  "items": [
    {
      "rank": 1,
      "item_id": "movie_18823",
      "title": "Clash of the Titans",
      "content_type": "movie",
      "score": 0.713,
      "reason": "Matches your Action, Adventure history"
    }
  ]
}
```

Unknown users receive HTTP 200 with `cold_start_discovery`. Invalid limits/types receive 422. `GET /healthz` is a
liveness check and `GET /readyz` confirms that an artifact is loaded.

## Tests and limitations

Unit tests cover EDA transformations, behavior validation, personalized/cold-start ranking, artifact round trips, and
temporal evaluation. The API integration test checks the JSON boundary.

This is a content recommender, not evidence about Netflix subscribers. Ratings are TMDb votes with severe vote-count
imbalance; the dataset is a fixed 1,000-title-per-year sample; catalog membership and regional availability are not
verified; descriptions are missing disproportionately for TV. Offline behavior metrics are meaningful only when the
provided event stream is genuine and includes exposure context.

## How to improve

1. Ingest consented impressions, play duration, completion, search, dismiss, and availability events with contracts.
2. Add multilingual sentence embeddings and an ANN index; compare by user-level temporal splits and catalog segments.
3. Learn a calibrated ranker from position-aware exposures, then validate with A/B tests and guardrails.
4. Add licensed maturity/territory data, deletion workflows, artifact signing, tracing, and staged promotion/rollback.
5. Monitor new-item coverage, language/genre parity, profile staleness, drift, p95 latency, and business outcomes.

