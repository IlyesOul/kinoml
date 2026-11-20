# Few-shot personalization: from cold-start problem to measured results

**An applied ML research project by Ilyes Ouldsaada** · MIT · Python / PyTorch / NumPy · [Interactive project report](./index.html)

> **Question:** Can a recommender adapt to an unseen user's preferences from just **5, 10, or 20 ratings**? And do more complex meta-learning approaches actually beat strong, simple baselines?

I built an episodic evaluation pipeline for cold-start recommendation on **MovieLens-100K**. It learns movie representations from genre metadata, simulates new users with limited observed ratings, and compares **linear regression**, **similarity-based kNN**, a **matching-network-style predictor**, and **Reptile meta-learning**. The aim was to choose methods based on **measured ranking quality and complexity**, not just architectural novelty.

**[Open the interactive report](./index.html)** to explore the recorded results, model trade-offs, and limitations. The report can be deployed on GitHub Pages without a build step.

## At a glance

| What I owned | Implementation |
| --- | --- |
| Problem formulation | New-user recommendation with **5 / 10 / 20** observed ratings |
| Data | MovieLens-100K; filter users with at least 25 ratings; **user-disjoint** train/test split |
| Features | **19 genre indicators** from `u.item`, normalized and encoded with SimCLR-style self-supervision |
| Approaches | Per-user OLS, embedding-space kNN, matching-network-style attention, Reptile |
| Metric | Mean **NDCG** ranking 10 held-out **rated** movies per user |
| Diagnostics | Per-model mean / standard deviation, number of test episodes, training / inference timings |
| Software | Reproducible seeded experiment runner, CSV export, plotting, static report, validation checks |

## Recorded experiment: what happened?

The original run evaluated **164 held-out users** per model / support-size setting with **658 users** in the training pool. These are the *actual archived observations*, not newly generated numbers.

| Model | 5 ratings | 10 ratings | 20 ratings |
| --- | ---: | ---: | ---: |
| Linear regression | 0.8290 | 0.8220 | 0.8249 |
| kNN | 0.8233 | **0.8312** | 0.8283 |
| Matching network | 0.8320 | 0.8295 | 0.8176 |
| Reptile | **0.8437** | 0.8275 | **0.8460** |

*Metric: mean NDCG over each user's held-out rated movies; larger is better. Scores are descriptive and **not an established performance uplift**.*

**Interpretation:** Reptile had the highest recorded score in the 5- and 20-rating settings, whereas a simpler kNN method led at 10. The most useful lesson was that added model complexity did **not consistently** improve results. I would want a fair, paired evaluation with repeated seeds and uncertainty estimates before recommending any model for deployment.

### Important experimental caveat — read before comparing the numbers

The **archived CSV was produced with a flaw in comparative evaluation**: each model received a different seeded random sample of support/query ratings for the held-out users. Therefore, the table cannot support a controlled claim that Reptile *outperforms* another model. I identified and corrected this in `methods.py`: **all models now use the same evaluation seed / per-user episodes for a given support size**. The updated experiment must be rerun on MovieLens-100K before publishing paired results. **I have not relabeled the original observations as rerun results.** See [Experiment design and limitations](docs/experiment_design.md).

This distinction matters for applied science: a result is only as persuasive as the experimental comparison behind it.

## Why these methods?

1. **Simple baselines first:** Per-episode least squares measures how far a linear signal can go; kNN tests whether proximity in embedding space is already sufficient.
2. **Representations without rating labels:** A SimCLR-inspired objective maps movie **genre vectors** to embeddings using random masks, dropout, and noise; this is a practical test of unsupervised representations, **not** a pretrained text/vision model.
3. **Learn to adapt:** A matching-network-style model predicts ratings through attention to a user's support ratings. Reptile learns an initialization for rapid gradient updates on a new user.
4. **Evaluate across data scarcity:** Vary the support set size while reserving query items and holding out test users entirely from supervised meta-training.

## Reproduce the experiment

This repo includes the original source and archived results, but **not the underlying MovieLens dataset**. For a full rerun:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt

# Download MovieLens-100K from https://grouplens.org/datasets/movielens/100k/
# Extract so that ./ml-100k/u.data and ./ml-100k/u.item exist.
python methods.py --no-plots
python scripts/validate_results.py experiment_results.csv
python -m unittest discover -s tests -v
```

`python methods.py` also saves diagnostic plots to `plots/`. The full run trains several models for each support size, so it is substantially slower than running the tests. The result CSV is overwritten on rerun; **commit it only once you have validated the new protocol and numbers**. The project page reads the checked-in CSV, so updated results display automatically when served by GitHub Pages.

### Test / project report locally

```bash
python -m http.server 8000
# Open http://localhost:8000
```

The site has no CDN or external JavaScript dependencies. The included CSV provides the project’s historical results out of the box. To publish, upload the **contents of this folder** (not the outer folder) to a GitHub repository, then configure GitHub Pages to serve the repository's root. Use the resulting page or repo URL for the application.

## Repository guide

```text
index.html                       Interactive research report (GitHub Pages)
methods.py                       Dataset, models, episodic training and evaluation
experiment_results.csv           Original 12-row experimental record (unpaired)
scripts/validate_results.py      Result-schema / sanity checks (stdlib only)
tests/                          Deterministic metric, split, episode, CSV tests
docs/experiment_design.md        Protocol, limitations and next experiments
docs/original_research_proposal.pdf  Initial study proposal (not a results paper)
requirements.txt                 Python dependencies for model reruns
```

## Next experiments I would prioritize

- **Rerun the corrected paired protocol** on multiple random seeds; report confidence intervals on per-user paired score differences.
- Include an **identity-genre-feature baseline** to isolate the value of contrastive pretraining.
- Add more realistic candidate generation with unrated items / negative sampling, and evaluate on a **temporal split** to model the arrival of genuinely new preferences.
- Profile inference costs and test whether the performance/complexity trade-off makes sense for an actual service.

### Why I chose to share this for an applied science role

The most transferable part of this project isn't movies: it's translating an ambiguous personalization question into a measurable ML problem, preparing data, choosing a simple comparison, iterating on alternative approaches, and being candid about what the evidence does and does not establish. This is the approach I'd bring to product, risk, and decisioning problems where correctness and useful impact matter more than a benchmark headline.

---

**Data:** [GroupLens MovieLens-100K](https://grouplens.org/datasets/movielens/100k/) (see the data publisher's licensing terms). **Author:** Ilyes Ouldsaada. This is an independent educational / research prototype, not a production recommender or a Ramp project.
