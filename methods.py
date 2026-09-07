"""Few-shot recommendation experiment for MovieLens-100K.

Implements the methods/evaluation described in the project PDF:
- Episodic few-shot setup with support/query sets per user
- Evaluation with user-disjoint train/test split
- NDCG as the primary metric
- Baselines: linear regression on item features (per episode) and kNN on embeddings
- Meta-learning approaches: matching-network style model and Reptile
- Item embeddings: SimCLR-style self-supervised learning over movie feature vectors
"""

from __future__ import annotations

import random
import time
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset


DATA_DIR = Path("ml-100k")
DATA_FILE = DATA_DIR / "data.txt"
ITEM_FILE = DATA_DIR / "u.item"
DEFAULT_SUPPORT_SIZES = (5, 10, 20)
DEFAULT_TRAIN_USER_FRACTIONS = (0.25, 0.5, 0.75, 1.0)
DEFAULT_RESULTS_PATH = Path("experiment_results.csv")
DEFAULT_PLOTS_DIR = Path("plots")
DEFAULT_SIMCLR_EMBED_DIM = 32
MODEL_ORDER = (
    "linear_regression",
    "knn",
    "matching_network",
    "reptile",
)


@dataclass
class Interaction:
    user_id: int
    item_id: int
    rating: float
    timestamp: int


@dataclass
class Episode:
    user_id: int
    support_item_ids: np.ndarray
    support_ratings: np.ndarray
    query_item_ids: np.ndarray
    query_ratings: np.ndarray


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_interactions(path: Path = DATA_FILE) -> List[Interaction]:
    interactions: List[Interaction] = []
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.strip().split("\t")
            if len(parts) != 4:
                continue
            user_id, item_id, rating, timestamp = map(int, parts)
            interactions.append(
                Interaction(
                    user_id=user_id,
                    item_id=item_id,
                    rating=float(rating),
                    timestamp=timestamp,
                )
            )
    return interactions


def load_item_genre_embeddings(path: Path = ITEM_FILE) -> Dict[int, np.ndarray]:
    item_to_vec: Dict[int, np.ndarray] = {}
    with path.open("r", encoding="latin-1") as f:
        for line in f:
            parts = line.rstrip("\n").split("|")
            # u.item: first 5 metadata fields + 19 genre flags.
            if len(parts) < 24:
                continue
            item_id = int(parts[0])
            genre_vec = np.array([float(x) for x in parts[-19:]], dtype=np.float32)
            if genre_vec.sum() > 0:
                genre_vec = genre_vec / genre_vec.sum()
            item_to_vec[item_id] = genre_vec
    return item_to_vec


def _augment_features(
    x: torch.Tensor,
    mask_prob: float = 0.15,
    dropout_prob: float = 0.15,
    noise_std: float = 0.05,
) -> torch.Tensor:
    """Random feature masking/dropout/noise for SimCLR views."""
    out = x.clone()
    if mask_prob > 0:
        mask = torch.rand_like(out) < mask_prob
        out = out.masked_fill(mask, 0.0)
    if dropout_prob > 0:
        keep = (torch.rand_like(out) >= dropout_prob).float()
        out = out * keep
    if noise_std > 0:
        out = out + noise_std * torch.randn_like(out)
    return out


def _nt_xent_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float = 0.2) -> torch.Tensor:
    """NT-Xent contrastive loss for two normalized view batches."""
    bs = z1.size(0)
    z = torch.cat([z1, z2], dim=0)
    sim = z @ z.T
    sim = sim / temperature

    eye = torch.eye(2 * bs, device=z.device, dtype=torch.bool)
    sim = sim.masked_fill(eye, -1e9)

    targets = torch.arange(bs, device=z.device)
    targets = torch.cat([targets + bs, targets], dim=0)
    return F.cross_entropy(sim, targets)


class SimCLREncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, embedding_dim: int = DEFAULT_SIMCLR_EMBED_DIM):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.projector = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.encode(x)
        return self.projector(h)


def train_simclr_item_embeddings(
    item_features: Dict[int, np.ndarray],
    embedding_dim: int = DEFAULT_SIMCLR_EMBED_DIM,
    hidden_dim: int = 64,
    epochs: int = 80,
    batch_size: int = 256,
    lr: float = 1e-3,
    temperature: float = 0.2,
    mask_prob: float = 0.15,
    dropout_prob: float = 0.15,
    noise_std: float = 0.05,
    device: torch.device | None = None,
    seed: int = 42,
) -> Dict[int, np.ndarray]:
    """Learn item embeddings with SimCLR loop over item feature vectors."""
    rng = np.random.default_rng(seed)
    device = device or torch.device("cpu")
    item_ids = sorted(item_features.keys())
    x_np = np.stack([item_features[i] for i in item_ids], axis=0).astype(np.float32)

    # Slight shuffle to avoid fixed ordering bias in small datasets.
    perm = rng.permutation(len(item_ids))
    x_np = x_np[perm]
    item_ids = [item_ids[i] for i in perm]

    x = torch.tensor(x_np, dtype=torch.float32)
    dataset = TensorDataset(x)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True, drop_last=False)

    model = SimCLREncoder(input_dim=x.shape[1], hidden_dim=hidden_dim, embedding_dim=embedding_dim).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    model.train()

    for _ in range(epochs):
        for (xb,) in loader:
            xb = xb.to(device)
            # SimCLR format:
            # x1 = augment(x), x2 = augment(x)
            x1 = _augment_features(xb, mask_prob=mask_prob, dropout_prob=dropout_prob, noise_std=noise_std)
            x2 = _augment_features(xb, mask_prob=mask_prob, dropout_prob=dropout_prob, noise_std=noise_std)

            # z1 = model(x1), z2 = model(x2)
            z1 = model(x1)
            z2 = model(x2)

            # normalize + contrastive loss
            z1 = F.normalize(z1, dim=-1)
            z2 = F.normalize(z2, dim=-1)
            loss = _nt_xent_loss(z1, z2, temperature=temperature)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        full_x = torch.tensor(x_np, dtype=torch.float32, device=device)
        emb = model.encode(full_x)
        emb = F.normalize(emb, dim=-1).cpu().numpy().astype(np.float32)
    return {item_id: emb[idx] for idx, item_id in enumerate(item_ids)}


def build_user_histories(
    interactions: Sequence[Interaction],
) -> Dict[int, List[Tuple[int, float, int]]]:
    histories: Dict[int, List[Tuple[int, float, int]]] = {}
    for inter in interactions:
        histories.setdefault(inter.user_id, []).append(
            (inter.item_id, inter.rating, inter.timestamp)
        )
    for user_id in histories:
        histories[user_id].sort(key=lambda x: x[2])
    return histories


def train_test_user_split(
    user_histories: Dict[int, List[Tuple[int, float, int]]],
    test_ratio: float = 0.2,
    min_ratings: int = 25,
    seed: int = 42,
) -> Tuple[List[int], List[int]]:
    eligible_users = [
        u for u, hist in user_histories.items() if len(hist) >= min_ratings
    ]
    rng = random.Random(seed)
    rng.shuffle(eligible_users)
    n_test = max(1, int(len(eligible_users) * test_ratio))
    test_users = eligible_users[:n_test]
    train_users = eligible_users[n_test:]
    return train_users, test_users


def make_episode_for_user(
    user_id: int,
    user_histories: Dict[int, List[Tuple[int, float, int]]],
    support_size: int,
    query_size: int = 10,
    rng: random.Random | None = None,
) -> Episode | None:
    rng = rng or random.Random()
    hist = user_histories[user_id]
    if len(hist) < support_size + 1:
        return None
    idx = list(range(len(hist)))
    rng.shuffle(idx)
    support_idx = idx[:support_size]
    remaining = idx[support_size:]
    if not remaining:
        return None
    q_size = min(query_size, len(remaining))
    query_idx = remaining[:q_size]

    s_items = np.array([hist[i][0] for i in support_idx], dtype=np.int64)
    s_ratings = np.array([hist[i][1] for i in support_idx], dtype=np.float32)
    q_items = np.array([hist[i][0] for i in query_idx], dtype=np.int64)
    q_ratings = np.array([hist[i][1] for i in query_idx], dtype=np.float32)
    return Episode(
        user_id=user_id,
        support_item_ids=s_items,
        support_ratings=s_ratings,
        query_item_ids=q_items,
        query_ratings=q_ratings,
    )


def dcg_at_k(relevance: np.ndarray, k: int | None = None) -> float:
    if relevance.size == 0:
        return 0.0
    if k is not None:
        relevance = relevance[:k]
    discounts = np.log2(np.arange(2, relevance.size + 2))
    return float(np.sum((2**relevance - 1) / discounts))


def ndcg_from_scores(
    y_true: np.ndarray, y_score: np.ndarray, k: int | None = None
) -> float:
    if y_true.size == 0:
        return 0.0
    order = np.argsort(-y_score)
    ranked_true = y_true[order]
    dcg = dcg_at_k(ranked_true, k=k)
    ideal_true = np.sort(y_true)[::-1]
    idcg = dcg_at_k(ideal_true, k=k)
    if idcg == 0:
        return 0.0
    return dcg / idcg


class LinearRegressionBaseline:
    """Ordinary least squares on item embeddings, fit fresh on each episode's support set."""

    def fit(self, train_interactions: Sequence[Interaction]) -> None:
        _ = train_interactions

    def predict_episode(
        self, episode: Episode, item_embeddings: Dict[int, np.ndarray]
    ) -> np.ndarray:
        x_s = np.stack([item_embeddings[i] for i in episode.support_item_ids], axis=0).astype(
            np.float64
        )
        y_s = episode.support_ratings.astype(np.float64)
        x_q = np.stack([item_embeddings[i] for i in episode.query_item_ids], axis=0).astype(
            np.float64
        )
        ones_s = np.ones((x_s.shape[0], 1), dtype=np.float64)
        ones_q = np.ones((x_q.shape[0], 1), dtype=np.float64)
        x_s_b = np.hstack([x_s, ones_s])
        x_q_b = np.hstack([x_q, ones_q])
        if x_s_b.shape[0] < 1:
            return np.full(x_q.shape[0], float(np.mean(y_s)), dtype=np.float32)
        coef, _, rank, _ = np.linalg.lstsq(x_s_b, y_s, rcond=None)
        if rank == 0 or not np.all(np.isfinite(coef)):
            pred = np.full(x_q.shape[0], float(np.mean(y_s)), dtype=np.float64)
        else:
            pred = x_q_b @ coef
        return pred.astype(np.float32)


class KNNBaseline:
    """kNN in item-embedding space: weight support ratings by cosine similarity to each query item."""

    def __init__(self, k: int = 5, eps: float = 1e-8) -> None:
        self.k = k
        self.eps = eps

    def fit(self, train_interactions: Sequence[Interaction]) -> None:
        _ = train_interactions

    def predict_episode(
        self, episode: Episode, item_embeddings: Dict[int, np.ndarray]
    ) -> np.ndarray:
        s_emb = np.stack([item_embeddings[i] for i in episode.support_item_ids], axis=0)
        s_r = episode.support_ratings
        q_emb = np.stack([item_embeddings[i] for i in episode.query_item_ids], axis=0)

        s_norm = s_emb / (np.linalg.norm(s_emb, axis=1, keepdims=True) + self.eps)
        q_norm = q_emb / (np.linalg.norm(q_emb, axis=1, keepdims=True) + self.eps)
        sim = q_norm @ s_norm.T

        preds = np.zeros(q_emb.shape[0], dtype=np.float32)
        for qi in range(sim.shape[0]):
            nn_idx = np.argsort(-sim[qi])[: self.k]
            nn_sim = sim[qi, nn_idx]
            weights = np.maximum(nn_sim, 0.0)
            if weights.sum() < self.eps:
                preds[qi] = float(np.mean(s_r))
            else:
                preds[qi] = float(np.sum(weights * s_r[nn_idx]) / (weights.sum() + self.eps))
        return preds


class Encoder(nn.Module):
    """Simple MLP encoder used by meta-learning models."""

    def __init__(self, input_dim: int, hidden_dim: int, embedding_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MatchingNetworkRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64, emb_dim: int = 32) -> None:
        super().__init__()
        self.encoder = Encoder(input_dim, hidden_dim, emb_dim)

    def forward(
        self,
        support_x: torch.Tensor,
        support_y: torch.Tensor,
        query_x: torch.Tensor,
    ) -> torch.Tensor:
        s = F.normalize(self.encoder(support_x), dim=-1)
        q = F.normalize(self.encoder(query_x), dim=-1)
        attn = torch.softmax(q @ s.T, dim=-1)
        return attn @ support_y


class ReptileRegressor(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int = 64) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _episode_to_tensors(
    episode: Episode, item_embeddings: Dict[int, np.ndarray], device: torch.device
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    s_x = torch.tensor(
        np.stack([item_embeddings[i] for i in episode.support_item_ids], axis=0),
        dtype=torch.float32,
        device=device,
    )
    s_y = torch.tensor(episode.support_ratings, dtype=torch.float32, device=device)
    q_x = torch.tensor(
        np.stack([item_embeddings[i] for i in episode.query_item_ids], axis=0),
        dtype=torch.float32,
        device=device,
    )
    q_y = torch.tensor(episode.query_ratings, dtype=torch.float32, device=device)
    return s_x, s_y, q_x, q_y


def train_matching_network(
    model: MatchingNetworkRegressor,
    train_users: Sequence[int],
    user_histories: Dict[int, List[Tuple[int, float, int]]],
    item_embeddings: Dict[int, np.ndarray],
    support_size: int,
    num_episodes: int = 2000,
    query_size: int = 10,
    lr: float = 1e-3,
    device: torch.device | None = None,
    seed: int = 42,
) -> None:
    device = device or torch.device("cpu")
    model.to(device)
    model.train()
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    rng = random.Random(seed)
    for _ in range(num_episodes):
        user_id = rng.choice(train_users)
        ep = make_episode_for_user(
            user_id, user_histories, support_size=support_size, query_size=query_size, rng=rng
        )
        if ep is None:
            continue
        s_x, s_y, q_x, q_y = _episode_to_tensors(ep, item_embeddings, device)
        pred = model(s_x, s_y, q_x)
        loss = F.mse_loss(pred, q_y)
        opt.zero_grad()
        loss.backward()
        opt.step()


def train_reptile(
    model: ReptileRegressor,
    train_users: Sequence[int],
    user_histories: Dict[int, List[Tuple[int, float, int]]],
    item_embeddings: Dict[int, np.ndarray],
    support_size: int,
    meta_iterations: int = 1200,
    inner_steps: int = 5,
    inner_lr: float = 1e-2,
    meta_step_size: float = 0.1,
    query_size: int = 10,
    device: torch.device | None = None,
    seed: int = 42,
) -> None:
    device = device or torch.device("cpu")
    model.to(device)
    model.train()
    rng = random.Random(seed)

    for _ in range(meta_iterations):
        user_id = rng.choice(train_users)
        ep = make_episode_for_user(
            user_id, user_histories, support_size=support_size, query_size=query_size, rng=rng
        )
        if ep is None:
            continue
        s_x, s_y, _, _ = _episode_to_tensors(ep, item_embeddings, device)

        fast_model = deepcopy(model)
        fast_model.to(device)
        fast_opt = torch.optim.SGD(fast_model.parameters(), lr=inner_lr)
        for _ in range(inner_steps):
            pred_s = fast_model(s_x)
            inner_loss = F.mse_loss(pred_s, s_y)
            fast_opt.zero_grad()
            inner_loss.backward()
            fast_opt.step()

        with torch.no_grad():
            for p, fp in zip(model.parameters(), fast_model.parameters()):
                p.data.add_(meta_step_size * (fp.data - p.data))


def evaluate_model(
    model_name: str,
    predictor,
    test_users: Sequence[int],
    user_histories: Dict[int, List[Tuple[int, float, int]]],
    item_embeddings: Dict[int, np.ndarray],
    support_size: int,
    query_size: int = 10,
    ndcg_k: int | None = None,
    seed: int = 123,
) -> Dict[str, float]:
    rng = random.Random(seed)
    ndcgs = []
    inference_start = time.perf_counter()

    for user_id in test_users:
        ep = make_episode_for_user(
            user_id, user_histories, support_size=support_size, query_size=query_size, rng=rng
        )
        if ep is None:
            continue
        pred = predictor(ep, item_embeddings)
        score = ndcg_from_scores(ep.query_ratings, pred, k=ndcg_k)
        ndcgs.append(score)

    inference_time = time.perf_counter() - inference_start
    return {
        "model": model_name,
        "support_size": float(support_size),
        "ndcg_mean": float(np.mean(ndcgs)) if ndcgs else 0.0,
        "ndcg_std": float(np.std(ndcgs)) if ndcgs else 0.0,
        "n_episodes": float(len(ndcgs)),
        "inference_time_sec": float(inference_time),
    }


def run_experiment(
    support_sizes: Sequence[int] = DEFAULT_SUPPORT_SIZES,
    test_ratio: float = 0.2,
    train_user_fraction: float = 1.0,
    min_user_ratings: int = 25,
    query_size: int = 10,
    seed: int = 42,
) -> List[Dict[str, float]]:
    set_seed(seed)
    interactions = load_interactions()
    item_features = load_item_genre_embeddings()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    simclr_start = time.perf_counter()
    item_embeddings = train_simclr_item_embeddings(
        item_features=item_features,
        embedding_dim=DEFAULT_SIMCLR_EMBED_DIM,
        hidden_dim=64,
        epochs=80,
        batch_size=256,
        lr=1e-3,
        temperature=0.2,
        mask_prob=0.15,
        dropout_prob=0.15,
        noise_std=0.05,
        device=device,
        seed=seed,
    )
    simclr_train_time = time.perf_counter() - simclr_start
    user_histories = build_user_histories(interactions)
    train_users, test_users = train_test_user_split(
        user_histories, test_ratio=test_ratio, min_ratings=min_user_ratings, seed=seed
    )
    train_pool_size = len(train_users)
    frac = float(min(1.0, max(1e-9, train_user_fraction)))
    n_keep = max(1, int(train_pool_size * frac))
    train_users = train_users[:n_keep]
    n_train_users = len(train_users)

    emb_dim = next(iter(item_embeddings.values())).shape[0]
    results: List[Dict[str, float]] = []

    for support_size in support_sizes:
        # Baseline: linear regression (per-episode on support)
        lin = LinearRegressionBaseline()
        t0 = time.perf_counter()
        lin.fit([])
        train_t = time.perf_counter() - t0
        lin_result = evaluate_model(
            "linear_regression",
            lin.predict_episode,
            test_users,
            user_histories,
            item_embeddings,
            support_size=support_size,
            query_size=query_size,
            ndcg_k=None,
            seed=seed + support_size,
        )
        lin_result["train_time_sec"] = float(train_t)
        lin_result["train_user_fraction"] = float(frac)
        lin_result["n_train_users"] = float(n_train_users)
        lin_result["train_pool_size"] = float(train_pool_size)
        lin_result["simclr_train_time_sec"] = float(simclr_train_time)
        results.append(lin_result)

        # Baseline: kNN on embeddings
        knn = KNNBaseline(k=5)
        t0 = time.perf_counter()
        knn.fit([])
        train_t = time.perf_counter() - t0
        knn_result = evaluate_model(
            "knn",
            knn.predict_episode,
            test_users,
            user_histories,
            item_embeddings,
            support_size=support_size,
            query_size=query_size,
            ndcg_k=None,
            seed=seed + support_size + 1,
        )
        knn_result["train_time_sec"] = float(train_t)
        knn_result["train_user_fraction"] = float(frac)
        knn_result["n_train_users"] = float(n_train_users)
        knn_result["train_pool_size"] = float(train_pool_size)
        knn_result["simclr_train_time_sec"] = float(simclr_train_time)
        results.append(knn_result)

        # Matching-network style model
        metric_model = MatchingNetworkRegressor(input_dim=emb_dim, hidden_dim=64, emb_dim=32)
        t0 = time.perf_counter()
        train_matching_network(
            metric_model,
            train_users=train_users,
            user_histories=user_histories,
            item_embeddings=item_embeddings,
            support_size=support_size,
            num_episodes=1200,
            query_size=query_size,
            lr=1e-3,
            device=device,
            seed=seed + 11 + support_size,
        )
        metric_train_t = time.perf_counter() - t0

        def _metric_predict(ep: Episode, emb: Dict[int, np.ndarray]) -> np.ndarray:
            metric_model.eval()
            with torch.no_grad():
                s_x, s_y, q_x, _ = _episode_to_tensors(ep, emb, device)
                p = metric_model(s_x, s_y, q_x).detach().cpu().numpy().astype(np.float32)
            return p

        metric_result = evaluate_model(
            "matching_network",
            _metric_predict,
            test_users,
            user_histories,
            item_embeddings,
            support_size=support_size,
            query_size=query_size,
            ndcg_k=None,
            seed=seed + support_size + 2,
        )
        metric_result["train_time_sec"] = float(metric_train_t)
        metric_result["train_user_fraction"] = float(frac)
        metric_result["n_train_users"] = float(n_train_users)
        metric_result["train_pool_size"] = float(train_pool_size)
        metric_result["simclr_train_time_sec"] = float(simclr_train_time)
        results.append(metric_result)

        # Reptile model
        reptile = ReptileRegressor(input_dim=emb_dim, hidden_dim=64)
        t0 = time.perf_counter()
        train_reptile(
            reptile,
            train_users=train_users,
            user_histories=user_histories,
            item_embeddings=item_embeddings,
            support_size=support_size,
            meta_iterations=800,
            inner_steps=5,
            inner_lr=1e-2,
            meta_step_size=0.1,
            query_size=query_size,
            device=device,
            seed=seed + 21 + support_size,
        )
        reptile_train_t = time.perf_counter() - t0

        def _reptile_predict(ep: Episode, emb: Dict[int, np.ndarray]) -> np.ndarray:
            reptile.eval()
            with torch.no_grad():
                s_x, s_y, q_x, _ = _episode_to_tensors(ep, emb, device)

            # Task-specific adaptation at inference time.
            adapted = deepcopy(reptile)
            adapted.to(device)
            adapted.train()
            opt = torch.optim.SGD(adapted.parameters(), lr=1e-2)
            for _ in range(5):
                pred_s = adapted(s_x)
                loss_s = F.mse_loss(pred_s, s_y)
                opt.zero_grad()
                loss_s.backward()
                opt.step()
            adapted.eval()
            with torch.no_grad():
                p = adapted(q_x).detach().cpu().numpy().astype(np.float32)
            return p

        reptile_result = evaluate_model(
            "reptile",
            _reptile_predict,
            test_users,
            user_histories,
            item_embeddings,
            support_size=support_size,
            query_size=query_size,
            ndcg_k=None,
            seed=seed + support_size + 3,
        )
        reptile_result["train_time_sec"] = float(reptile_train_t)
        reptile_result["train_user_fraction"] = float(frac)
        reptile_result["n_train_users"] = float(n_train_users)
        reptile_result["train_pool_size"] = float(train_pool_size)
        reptile_result["simclr_train_time_sec"] = float(simclr_train_time)
        results.append(reptile_result)

    return results


def run_experiment_grid(
    support_sizes: Sequence[int] = DEFAULT_SUPPORT_SIZES,
    train_user_fractions: Sequence[float] = DEFAULT_TRAIN_USER_FRACTIONS,
    test_ratio: float = 0.2,
    min_user_ratings: int = 25,
    query_size: int = 10,
    seed: int = 42,
) -> List[Dict[str, float]]:
    """Run `run_experiment` for each train-user fraction (nested prefix of the same shuffled pool)."""
    all_rows: List[Dict[str, float]] = []
    for tf in train_user_fractions:
        # Same seed for every fraction so train users are nested prefixes of one shuffled pool.
        rows = run_experiment(
            support_sizes=support_sizes,
            test_ratio=test_ratio,
            train_user_fraction=tf,
            min_user_ratings=min_user_ratings,
            query_size=query_size,
            seed=seed,
        )
        all_rows.extend(rows)
    return all_rows


def format_results_csv(results: Sequence[Dict[str, float]]) -> str:
    by_key = sorted(
        results,
        key=lambda x: (
            x.get("train_user_fraction", 1.0),
            x["support_size"],
            x["model"],
        ),
    )
    lines = [
        "model,support_size,train_user_fraction,n_train_users,train_pool_size,"
        "ndcg_mean,ndcg_std,n_episodes,train_time_sec,inference_time_sec,simclr_train_time_sec"
    ]
    for row in by_key:
        tf = row.get("train_user_fraction", 1.0)
        n_tr = int(row.get("n_train_users", 0))
        pool = int(row.get("train_pool_size", 0))
        lines.append(
            f"{row['model']},"
            f"{int(row['support_size'])},"
            f"{float(tf):.4f},"
            f"{n_tr},"
            f"{pool},"
            f"{row['ndcg_mean']:.4f},"
            f"{row['ndcg_std']:.4f},"
            f"{int(row['n_episodes'])},"
            f"{row.get('train_time_sec', 0.0):.3f},"
            f"{row['inference_time_sec']:.3f},"
            f"{row.get('simclr_train_time_sec', 0.0):.3f}"
        )
    return "\n".join(lines) + "\n"


def print_results(results: Sequence[Dict[str, float]]) -> None:
    print(format_results_csv(results).rstrip())


def write_results_to_file(
    results: Sequence[Dict[str, float]], path: Path | str = DEFAULT_RESULTS_PATH
) -> Path:
    out = Path(path)
    out.write_text(format_results_csv(results), encoding="utf-8")
    return out


def _ordered_models_present(results: Sequence[Dict[str, float]]) -> List[str]:
    seen = {r["model"] for r in results}
    return [m for m in MODEL_ORDER if m in seen] + sorted(seen.difference(MODEL_ORDER))


def _lookup_row(
    results: Sequence[Dict[str, float]],
    model: str,
    support_size: int,
    train_user_fraction: float,
) -> Optional[Dict[str, float]]:
    for r in results:
        if (
            r["model"] == model
            and int(r["support_size"]) == support_size
            and abs(float(r.get("train_user_fraction", 1.0)) - train_user_fraction) < 1e-6
        ):
            return r
    return None


def plot_experiment_results(
    results: Sequence[Dict[str, float]],
    out_dir: Path | str = DEFAULT_PLOTS_DIR,
) -> List[Path]:
    """Save simple line plots: NDCG vs support size (full train pool), and vs train users if grid run."""
    import os

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    mpl_cache = out_path / ".matplotlib-cache"
    mpl_cache.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_cache.resolve()))

    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots. Install with: pip install matplotlib")
        return []
    saved: List[Path] = []

    train_fracs = sorted({float(r.get("train_user_fraction", 1.0)) for r in results})
    support_sizes = sorted({int(r["support_size"]) for r in results})
    models = _ordered_models_present(results)
    baselines = {"linear_regression", "knn"}
    # Few distinct colors, dashed baselines / solid meta-learners
    palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def _style(model: str, idx: int) -> Tuple[str, str]:
        ls = "--" if model in baselines else "-"
        c = palette[idx % len(palette)]
        return ls, c

    # Use largest train fraction so curves reflect the full training pool when available.
    tf_plot = max(train_fracs) if train_fracs else 1.0
    n_tr_ref = next(
        (
            int(r.get("n_train_users", 0))
            for r in results
            if abs(float(r.get("train_user_fraction", 1.0)) - tf_plot) < 1e-6
        ),
        0,
    )
    pool_ref = next(
        (
            int(r.get("train_pool_size", 0))
            for r in results
            if abs(float(r.get("train_user_fraction", 1.0)) - tf_plot) < 1e-6
        ),
        0,
    )

    # --- Single plot: NDCG vs support set size ---
    fig1, ax1 = plt.subplots(figsize=(7.5, 4.5))
    for mi, model in enumerate(models):
        ys = []
        for s in support_sizes:
            row = _lookup_row(results, model, s, tf_plot)
            ys.append(row["ndcg_mean"] if row is not None else float("nan"))
        ls, c = _style(model, mi)
        ax1.plot(support_sizes, ys, linestyle=ls, color=c, marker="o", linewidth=2, markersize=7, label=model)
    ax1.set_xticks(support_sizes)
    ax1.set_xlabel("Support set size (movies per user episode)")
    ax1.set_ylabel("NDCG")
    title_extra = f" — training users {n_tr_ref}/{pool_ref}" if pool_ref else ""
    ax1.set_title(f"NDCG vs support size (train pool = {tf_plot:.0%}{title_extra})")
    ax1.set_ylim(0.0, 1.05)
    ax1.grid(True, linestyle=":", alpha=0.7)
    ax1.legend(frameon=True, loc="best", fontsize=9)
    fig1.text(
        0.5,
        0.01,
        "NDCG: ranking quality vs ideal order (true rating = relevance). 1 = best.",
        ha="center",
        fontsize=8,
        style="italic",
        transform=fig1.transFigure,
    )
    fig1.tight_layout()
    fig1.subplots_adjust(bottom=0.14)
    p1 = out_path / "ndcg_vs_support_size.png"
    fig1.savefig(p1, dpi=120, bbox_inches="tight")
    plt.close(fig1)
    saved.append(p1)

    # --- Optional second plot: NDCG vs # training users (one support size, single axes) ---
    if len(train_fracs) >= 2:
        s_mid = support_sizes[len(support_sizes) // 2]
        xs = sorted(
            {
                int(r.get("n_train_users", 0))
                for r in results
                if int(r["support_size"]) == s_mid
            }
        )
        fig2, ax2 = plt.subplots(figsize=(7.5, 4.5))
        for mi, model in enumerate(models):
            ys = []
            for n_tr in xs:
                row = next(
                    (
                        r
                        for r in results
                        if r["model"] == model
                        and int(r["support_size"]) == s_mid
                        and int(r.get("n_train_users", 0)) == n_tr
                    ),
                    None,
                )
                ys.append(row["ndcg_mean"] if row is not None else float("nan"))
            ls, c = _style(model, mi)
            ax2.plot(xs, ys, linestyle=ls, color=c, marker="s", linewidth=2, markersize=6, label=model)
        ax2.set_xlabel("Number of training users (meta-train pool)")
        ax2.set_ylabel("NDCG")
        ax2.set_title(f"NDCG vs training set size (support size = {s_mid})")
        ax2.set_ylim(0.0, 1.05)
        ax2.grid(True, linestyle=":", alpha=0.7)
        ax2.legend(frameon=True, loc="best", fontsize=9)
        fig2.text(
            0.5,
            0.01,
            "NDCG: ranking quality vs ideal order (true rating = relevance). 1 = best.",
            ha="center",
            fontsize=8,
            style="italic",
            transform=fig2.transFigure,
        )
        fig2.tight_layout()
        fig2.subplots_adjust(bottom=0.14)
        p2 = out_path / "ndcg_vs_train_users.png"
        fig2.savefig(p2, dpi=120, bbox_inches="tight")
        plt.close(fig2)
        saved.append(p2)

    return saved


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Few-shot recommendation experiment (MovieLens-100K).")
    parser.add_argument(
        "--train-fractions",
        type=float,
        nargs="*",
        default=None,
        metavar="F",
        help="Fractions of the training user pool to use (nested). Default: 1.0 only. Example: --train-fractions 0.25 0.5 0.75 1.0",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Do not write matplotlib figures to the plots directory.",
    )
    parser.add_argument(
        "--plots-dir",
        type=Path,
        default=DEFAULT_PLOTS_DIR,
        help=f"Directory for PNG figures (default: {DEFAULT_PLOTS_DIR}).",
    )
    args = parser.parse_args()
    train_fractions = args.train_fractions if args.train_fractions is not None else [1.0]

    start = time.perf_counter()
    if len(train_fractions) == 1:
        experiment_results = run_experiment(
            support_sizes=DEFAULT_SUPPORT_SIZES,
            test_ratio=0.2,
            train_user_fraction=train_fractions[0],
            min_user_ratings=25,
            query_size=10,
            seed=42,
        )
    else:
        experiment_results = run_experiment_grid(
            support_sizes=DEFAULT_SUPPORT_SIZES,
            train_user_fractions=tuple(train_fractions),
            test_ratio=0.2,
            min_user_ratings=25,
            query_size=10,
            seed=42,
        )

    out_path = write_results_to_file(experiment_results, DEFAULT_RESULTS_PATH)
    print_results(experiment_results)
    elapsed = time.perf_counter() - start
    print(f"\nResults written to {out_path.resolve()}")
    print(f"Total runtime: {elapsed:.2f} sec")

    if not args.no_plots:
        plot_paths = plot_experiment_results(experiment_results, out_dir=args.plots_dir)
        for p in plot_paths:
            print(f"Plot written to {p.resolve()}")
