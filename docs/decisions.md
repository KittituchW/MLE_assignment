# Engineering decisions

## ADR-001: online, single-machine assignment implementation

**Decision.** Serve per-user scores online from an in-memory bundle. Keep stages independently runnable and idempotent.

**Why.** The catalogue is only 32k items and the acceptance test is an online request. Spark, Databricks, Delta, or a
batch top-N table would add reviewer setup and failure modes without solving a scale problem. Batch top-N remains a
production fallback.

## ADR-002: simulated behavior behind an interface

**Decision.** Default to deterministic `SimulatedInteractionSource`; support `ParquetInteractionSource` with the exact
`user_id, item_id, event_ts, event_type, watch_ratio` contract.

**Alternatives.** Treat TMDb aggregate votes as user behavior, or build content-only recommendations. Both would
misrepresent the data and fail the stated behavior requirement. Simulator metrics are labeled as regression gates only.

## ADR-003: item-kNN instead of ALS

**Decision.** Use sparse item-kNN as the behavioral component.

**Why.** The supplied prototype found ALS within seed noise of item-kNN and below the hybrid. Item-kNN is small, fast,
explainable, and avoids a compiled recommender dependency. ALS is reconsidered when real implicit-feedback volume and
ranking gains justify it.

## ADR-004: content model for reach, not accuracy

**Decision.** TF-IDF title/genre/description features, 128-dimensional SVD, blended at 0.3.

**Why.** Content alone underperforms popularity but recommends items without training events. Sentence transformers and
Torch add large dependencies without measured value here.

## ADR-005: 0.6/0.3/0.1 hybrid

**Decision.** Per-request min-max normalization of item-kNN/content, blended with popularity at 0.6/0.3/0.1. These weights
were selected on the validation window and frozen before the final 14-day test.

**Alternative.** A learned ranker requires trustworthy real exposures and negatives; simulator labels would teach its own
rules more precisely without increasing evidence of user value.

## ADR-006: portable bundle separated from MLflow

**Decision.** Training logs/registers in MLflow, but serving reads only JSON, NPZ, and NPY. Promotion uses an atomic text
pointer and MLflow alias, not symlinks or stages.

**Why.** This keeps the API image and attack surface small, avoids pickle in serving, supports Windows, and allows a fully
validated hot swap with rollback. MLflow's CloudPickle warning is isolated to its non-serving registry wrapper.

## ADR-007: layered content heuristic

**Decision.** Keyword, low-evidence Romance/Drama movie, and curated item denylist layers run before recommendation. Manual
top-20 review added two titles.

**Why.** The source has no maturity flag and keyword-only approaches produced both misses and false positives. This is
explicitly not described as family-safe; production requires licensed maturity/rights metadata.

## ADR-008: temporal evaluation and conservative gate

**Decision.** Hold out the final 14 days, evaluate only new-to-user items, and promote only if Recall@10 and NDCG@10 are
both at least 1.5x popularity and no more than 5% below the champion.

**Why.** A random split leaks future behavior. Accuracy alone would hide catalogue collapse, so coverage and cold-item
share are first-class report fields.

## ADR-009: deterministic clocks and runtime paths

**Decision.** Dates come from config/arguments; pipeline logic never calls `datetime.now()`. Mutable files live under
`RECO_HOME`, outside the repository/OneDrive when configured.

**Why.** This makes reruns byte-stable, tests reproducible, and packaging clean. The CI grep guard prevents regression.

