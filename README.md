# Netflix content recommender

This repository is a self-contained online recommender built for the supplied ML engineering assignment. The source
catalogue has no user behavior, so interactions are **explicitly simulated** behind a replaceable interface. Offline
metrics are regression checks against that simulator, not evidence of real user impact.

The complete quickstart, captured API examples, design notes, and measured results are added after the verified full run.

## How does it work

The ordered pipeline validates and prepares the catalogue, simulates or reads interactions, trains a hybrid recommender,
evaluates it on a temporal holdout, promotes a bundle, and serves it through FastAPI.

## How to feed input and get output

Run `python -m reco serve`, then send `POST /v1/recommendations` with `{"user_id":"A"}`.

## How to improve in the future

Replace simulated events with consented production events first; model and scale improvements are documented separately.

