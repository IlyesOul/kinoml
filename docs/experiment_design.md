# Experiment design, validation and limitations

## Prediction target

Rank a small set of movies that a user has already rated, using their other ratings as a small adaptation/support set. Ratings (1–5) are treated as relevance labels for **NDCG**. This is a *ranking evaluation on observed-rated query movies*, **not** a test of full-catalog recommendation, missing-not-at-random exposure, conversion, business lift, or an online A/B experiment.

## Data and preprocessing

- Dataset: GroupLens MovieLens-100K, with `u.data` (user, movie, rating, timestamp) and `u.item` (metadata, including 19 genre flags).
- Feature pipeline: genre flags normalized per item, then a SimCLR-style learned projection (augmentation via stochastic masking, dropout and Gaussian noise); no textual movie descriptions or pre-trained language model.
- Supervised user split: filter users with >=25 interactions, shuffle with fixed seed, hold out 20% of eligible users. Recorded run: 658 training and 164 held-out test users (822 eligible).
- Episodes: for each user, randomly sample 5/10/20 support ratings and up to 10 different rated query movies. Meta-learners train on episodes from training users and predict for test users they did not encounter during supervised training.
- Item metadata/unsupervised representations span the movie catalog, so the split is **user-disjoint rather than item-disjoint**. No rating targets from test users are used in supervised meta-training, but the evaluation uses unlabeled item-level transductive information.

## Four model families

1. **Ordinary least squares** fits a new linear regression from each user's tiny labeled support set to their ratings on the encoded movie features.
2. **kNN** weights support-user ratings by cosine similarity to the queried movie embedding.
3. **Matching network** learns an embedding mapping and weighted attention from query to support items, with episodic supervised training.
4. **Reptile** meta-learns a network initialization, then fine-tunes on each test user's support labels with gradient steps before scoring the query set.

## What the original numbers support

`experiment_results.csv` is an archival record of a single run, including per-model NDCG means, standard deviations, 164 test episodes per configuration, and timings. It is suitable for an **exploratory description** of what was observed. It is **not a controlled model ranking**.

### Discovered flaw and correction

The previous code passed `seed + support_size + model_offset` into `evaluate_model`. Consequently each model saw **different support and query rating subsets per test user**, introducing variation unrelated to model quality. In the current `methods.py`, all models at a given support size instead share the exact same `evaluation_seed`. Since `evaluate_model` uses a locally seeded `random.Random` and visits test users in the same order, this makes the support/query episodes identical across models.

**This fix applies to future reruns; the archived CSV has NOT been regenerated.** The original dataset files were not part of the supplied repository materials, and no new empirical results or CIs have been invented.

## Remaining caveats

- **One seed / no uncertainty of pairwise differences:** A user-level mean ± standard deviation is not a confidence interval. Repeated splits or paired bootstrap resampling is needed before claiming an improvement.
- **Offline rating selection:** Only already-rated query movies are evaluated. This may be optimistic relative to recommending among unknown or never-seen items.
- **Random vs temporal:** Support/query assignments ignore chronology despite available timestamps. A temporal study is needed for a deployment-like cold-start scenario.
- **Imbalanced observable data:** Users with at least 25 ratings are eligible. This leaves out the true most-data-scarce users and biases the sample toward active raters.
- **No isolation of representation gains:** Without an untrained/identity feature baseline, the incremental contribution of the SimCLR-style embeddings cannot be attributed confidently.
- **No production or business metrics:** There was no system deployment, online randomized trial, fraud/credit problem, or causal inference analysis.
- **Timing is environment-dependent:** Train/inference seconds are diagnostic only and not portable latency guarantees. SimCLR time is common overhead and is reported separately.

## Proposed next validation protocol

1. Lock a data/feature contract and candidate generation policy.
2. Pre-generate test episodes per test user/support size/seed and persist episode IDs.
3. Evaluate all models on **identical episodes**; store per-user NDCG scores, not only aggregates.
4. Re-run over multiple user splits and model seeds; compute paired bootstrap intervals for per-user differences, with a prespecified primary comparison.
5. Add an identity-genre embedding ablation, popularity/mean prediction baseline, temporal split and inference-cost profile.
6. Only then consider the simplest model with credible gain and acceptable operational complexity.

This project is intentionally framed as a reproducible experimental artifact, not as evidence of already shipped business value.
