"""Even view sampling for cameras placed around an object, where frame order carries no meaning.

The cameras' directions from their centroid are partitioned into k-means regions; one random view is taken from
each region. Context views come from num_context_views regions over all cameras; target views from
num_target_views regions over the cameras that are left, so the two sets never overlap. The regions give
roughly even coverage of the viewing sphere and the random pick inside each region varies the views between
iterations.

Ported from TokenGS (tokengs/data/view_sampling.py and Provider._even_sample_indices), so both methods are
trained with the same view selection.
"""

from dataclasses import dataclass
from typing import Literal

import numpy as np
import torch
from jaxtyping import Float, Int64
from torch import Tensor

from .view_sampler import ViewSampler


@dataclass
class ViewSamplerEvenCfg:
    name: Literal["even"]
    num_context_views: int
    num_target_views: int
    seed: int  # seeds the k-means initialization only, so the regions are the same in every worker


def directions_from_c2ws(c2ws: np.ndarray) -> np.ndarray:
    """(N, 3) unit vectors from the centroid of the camera centers to each camera center."""
    centers = np.asarray(c2ws)[:, :3, 3].astype(np.float64)
    d = centers - centers.mean(axis=0, keepdims=True)
    n = np.linalg.norm(d, axis=1, keepdims=True)
    n = np.where(n < 1e-12, 1.0, n)  # guard against a degenerate (coincident) center
    return d / n


def kmeans_unit(dirs: np.ndarray, k: int, seed: int, n_iter: int = 50) -> np.ndarray:
    """Lloyd's k-means on unit vectors; returns (N,) labels in [0, k), every label non-empty.

    Deterministic given the seed: farthest-point initialization from a seeded first center, and first-index
    tie-breaks."""
    dirs = np.ascontiguousarray(dirs, dtype=np.float64)
    N = dirs.shape[0]
    assert 1 <= k <= N, f"kmeans_unit requires 1 <= k({k}) <= N({N})"
    if k == 1:
        return np.zeros(N, dtype=np.int64)
    if k == N:
        return np.arange(N, dtype=np.int64)

    rng = np.random.default_rng(seed)
    first = int(rng.integers(0, N))
    seed_idx = [first]
    d2 = np.sum((dirs - dirs[first]) ** 2, axis=1)  # min squared distance to the chosen centers
    for _ in range(1, k):
        nxt = int(np.argmax(d2))
        seed_idx.append(nxt)
        d2 = np.minimum(d2, np.sum((dirs - dirs[nxt]) ** 2, axis=1))
    centers = dirs[seed_idx].copy()

    labels = np.full(N, -1, dtype=np.int64)
    for _ in range(n_iter):
        dist = np.linalg.norm(dirs[:, None, :] - centers[None, :, :], axis=2)  # (N, k)
        new_labels = np.argmin(dist, axis=1)

        # Move each empty center onto the currently worst-served point, so all k clusters end up non-empty.
        for _repair in range(k):
            empty = np.where(np.bincount(new_labels, minlength=k) == 0)[0]
            if empty.size == 0:
                break
            served = dist[np.arange(N), new_labels].copy()
            for c in empty:
                worst = int(np.argmax(served))
                centers[c] = dirs[worst]
                served[worst] = -1.0  # don't reuse the same point for two empty clusters
            dist = np.linalg.norm(dirs[:, None, :] - centers[None, :, :], axis=2)
            new_labels = np.argmin(dist, axis=1)

        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        for c in range(k):
            members = dirs[labels == c]
            if members.shape[0] > 0:
                centers[c] = members.mean(axis=0)

    return labels.astype(np.int64)


def labels_to_members(labels: np.ndarray, k: int) -> list[np.ndarray]:
    return [np.flatnonzero(labels == c) for c in range(k)]


def pick_one_per_cluster(members: list[np.ndarray]) -> np.ndarray:
    # torch's RNG: the data loader seeds it differently in every worker and epoch.
    return np.array(
        [int(m[torch.randint(len(m), size=tuple()).item()]) for m in members],
        dtype=np.int64,
    )


class ViewSamplerEven(ViewSampler[ViewSamplerEvenCfg]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # Context regions per camera rig. Scenes rendered with the same rig share one entry.
        self._context_clusters: dict[bytes, list[np.ndarray]] = {}

    def sample(
        self,
        scene: str,
        extrinsics: Float[Tensor, "view 4 4"],
        intrinsics: Float[Tensor, "view 3 3"],
        device: torch.device = torch.device("cpu"),
        **kwargs,
    ) -> tuple[
        Int64[Tensor, " context_view"],  # indices for context views
        Int64[Tensor, " target_view"],  # indices for target views
    ]:
        k_context, k_target = self.cfg.num_context_views, self.cfg.num_target_views
        dirs = directions_from_c2ws(extrinsics.detach().cpu().numpy())
        num_views = dirs.shape[0]
        if num_views < k_context + k_target:
            raise ValueError(
                f"scene {scene} has {num_views} views; even sampling needs {k_context + k_target} "
                "for disjoint context and target views"
            )

        signature = np.round(dirs, 4).tobytes()
        members = self._context_clusters.get(signature)
        if members is None:
            members = labels_to_members(kmeans_unit(dirs, k_context, self.cfg.seed), k_context)
            self._context_clusters[signature] = members
        context = pick_one_per_cluster(members)

        leftover = np.setdiff1d(np.arange(num_views), context)
        target_members = labels_to_members(
            kmeans_unit(dirs[leftover], k_target, self.cfg.seed), k_target
        )
        target = leftover[pick_one_per_cluster(target_members)]

        return (
            torch.tensor(context, dtype=torch.int64, device=device),
            torch.tensor(target, dtype=torch.int64, device=device),
        )

    @property
    def num_context_views(self) -> int:
        return self.cfg.num_context_views

    @property
    def num_target_views(self) -> int:
        return self.cfg.num_target_views
