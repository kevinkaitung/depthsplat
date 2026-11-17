"""Volume-rendered scenes read directly from their NeRF-style transforms.json (no .torch chunks).

Layout: <root>/<train|test>/<scene>/gaussian_splat/{transforms.json, render_config.json, images/}. transforms.json
holds OpenGL camera-to-world matrices and pixel intrinsics; every camera looks at the volume center from one
distance r, and the renderer placed it at r = safety_margin * rho / tan(fov_y / 2), where rho is the volume's
half-diagonal (safety_margin and fov_y are in render_config.json). So the volume lies inside the ball
(center, rho).

With val_holdout > 0, each root instead holds the scenes directly (<root>/<scene>/gaussian_splat/...), as
TokenGS's and InSituNet's training folders do: the scenes of all roots are shuffled with holdout_seed, the first
val_holdout go to validation (and to the test stage) and the rest to training.

Normalization: the volume center is moved to the origin and positions are scaled by k / rho, so every volume
fits a ball of radius k whatever its world units. With bounds "tight", near / far are that ball's depth range from
any camera, k (r / rho -+ 1); with "fixed" they are cfg.near / cfg.far.

The geometry functions must stay identical to exps/ffgs_volume_scenes.py of the FFGS-benchmark repo, which
feeds the same scenes to this model at evaluation time.
"""

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from PIL import Image
from torch.utils.data import Dataset

from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler

GL_TO_CV = np.diag([1.0, -1.0, -1.0, 1.0])  # flips the camera's y and z axes (OpenGL / NeRF -> OpenCV)
SCENE_SUBDIR = "gaussian_splat"


@dataclass
class DatasetVolumeCfg(DatasetCfgCommon):
    name: Literal["volume"]
    roots: list[Path]
    augment: bool
    k: float  # radius of the volume's bounding ball in model units
    bounds: Literal["tight", "fixed"]
    near: float  # used when bounds is "fixed"
    far: float
    # 0: roots hold train/ and test/. > 0: roots hold the scenes directly, and this many are held out of
    # training for validation (see the module docstring)
    val_holdout: int = 0
    holdout_seed: int = 42


def list_scenes(directory: Path) -> list[Path]:
    """<directory>/<scene>/gaussian_splat for every scene with a transforms.json, sorted."""
    if not directory.is_dir():
        raise FileNotFoundError(f"{directory} does not exist")
    return sorted(
        path / SCENE_SUBDIR for path in directory.iterdir() if (path / SCENE_SUBDIR / "transforms.json").is_file()
    )


def holdout_split(scenes: list[Path], n_val: int, seed: int) -> tuple[list[Path], list[Path]]:
    """(train, val) scenes: a seeded shuffle puts n_val scenes in val; both lists keep the sorted order."""
    if n_val >= len(scenes):
        raise ValueError(f"val_holdout = {n_val} leaves no training scenes ({len(scenes)} scenes found)")
    shuffled = sorted(scenes)
    random.Random(seed).shuffle(shuffled)
    val = set(shuffled[:n_val])
    return [s for s in sorted(scenes) if s not in val], [s for s in sorted(scenes) if s in val]


def scene_geometry(transforms: dict, render_cfg: dict, radius_tol: float = 1e-3) -> dict:
    """Volume center c, camera distance r and volume half-diagonal rho, from all of the scene's cameras."""
    c2w = np.array([f["transform_matrix"] for f in transforms["frames"]], dtype=np.float64)  # OpenGL c2w
    pos, fwd = c2w[:, :3, 3], -c2w[:, :3, 2]
    fwd = fwd / np.linalg.norm(fwd, axis=1, keepdims=True)
    # least-squares point closest to every optical axis
    proj = np.eye(3)[None] - fwd[:, :, None] * fwd[:, None, :]
    center = np.linalg.solve(proj.sum(0), np.einsum("nij,nj->i", proj, pos))
    dist = np.linalg.norm(pos - center, axis=1)
    r = float(dist.mean())
    if dist.std() / r > radius_tol:
        raise ValueError(
            f"cameras are not at one distance from the look-at point (r = {r:.3f} +- {dist.std():.3f}); "
            "the volume-anchored normalization assumes the renderer's orbit cameras"
        )
    margin, fov_y = render_cfg.get("safety_margin"), render_cfg.get("fov_y")
    if margin is None or fov_y is None:
        raise ValueError("render_config.json lacks safety_margin / fov_y")
    rho = r * math.tan(math.radians(fov_y) / 2) / margin
    return {"center": center, "r": r, "rho": float(rho)}


def normalized_cameras(transforms: dict, geom: dict, k: float) -> tuple[torch.Tensor, torch.Tensor]:
    """All frames' OpenCV c2w after moving the volume center to the origin and scaling by k / rho, plus the
    normalized intrinsics (fx / W, fy / H, cx / W, cy / H)."""
    c2w = np.array([f["transform_matrix"] for f in transforms["frames"]], dtype=np.float64)
    c2w = c2w @ GL_TO_CV
    c2w[:, :3, 3] = (c2w[:, :3, 3] - geom["center"]) * (k / geom["rho"])
    w, h = transforms["w"], transforms["h"]
    K = np.array(
        [
            [transforms["fl_x"] / w, 0, transforms["cx"] / w],
            [0, transforms["fl_y"] / h, transforms["cy"] / h],
            [0, 0, 1],
        ],
        dtype=np.float64,
    )
    return torch.from_numpy(c2w).float(), torch.from_numpy(K).float()


def tight_bounds(geom: dict, k: float) -> tuple[float, float]:
    """Depth range of the volume's bounding ball from any camera, in normalized units."""
    ratio = geom["r"] / geom["rho"]
    return k * (ratio - 1.0), k * (ratio + 1.0)


class DatasetVolume(Dataset):
    cfg: DatasetVolumeCfg
    stage: Stage
    view_sampler: ViewSampler
    scenes: list[Path]  # <root>/<stage>/<scene>/gaussian_splat

    def __init__(
        self,
        cfg: DatasetVolumeCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()
        self._cameras: dict[int, dict] = {}  # per-process cache; a scene's cameras never change

        # roots hold the scenes directly when held-out validation is on; the test stage then gets the held-out scenes
        flat = cfg.val_holdout > 0
        scene_dirs = [root if flat else root / self.data_stage for root in cfg.roots]
        self.scenes = [scene for directory in scene_dirs for scene in list_scenes(directory)]
        if cfg.overfit_to_scene is not None:
            self.scenes = [s for s in self.scenes if s.parent.name == cfg.overfit_to_scene]
        elif flat:
            train, val = holdout_split(self.scenes, cfg.val_holdout, cfg.holdout_seed)
            self.scenes = train if stage == "train" else val
        if not self.scenes:
            raise FileNotFoundError(
                f"no scenes with {SCENE_SUBDIR}/transforms.json under {[str(d) for d in scene_dirs]}"
            )

    @property
    def data_stage(self) -> Stage:
        # As the other datasets: validation shows scenes of the test split.
        if self.cfg.overfit_to_scene is not None:
            return "train"
        if self.stage == "val":
            return "test"
        return self.stage

    def load_cameras(self, index: int) -> dict:
        cameras = self._cameras.get(index)
        if cameras is None:
            scene_dir = self.scenes[index]
            with (scene_dir / "transforms.json").open("r") as f:
                transforms = json.load(f)
            with (scene_dir / "render_config.json").open("r") as f:
                render_cfg = json.load(f)
            geom = scene_geometry(transforms, render_cfg)
            extrinsics, K = normalized_cameras(transforms, geom, self.cfg.k)
            if self.cfg.bounds == "tight":
                near, far = tight_bounds(geom, self.cfg.k)
            else:
                near, far = self.cfg.near, self.cfg.far
            cameras = {
                "extrinsics": extrinsics,
                "intrinsics": K[None].repeat(len(extrinsics), 1, 1),
                "near": near,
                "far": far,
                "files": [frame["file_path"] for frame in transforms["frames"]],
            }
            self._cameras[index] = cameras
        return cameras

    def load_images(self, scene_dir: Path, files: list[str], indices: torch.Tensor) -> torch.Tensor:
        images = []
        for i in indices.tolist():
            with Image.open(scene_dir / files[i]) as image:
                images.append(self.to_tensor(image.convert("RGB")))
        return torch.stack(images)

    def __getitem__(self, index: int):
        scene_dir = self.scenes[index]
        scene = scene_dir.parent.name
        cameras = self.load_cameras(index)
        extrinsics, intrinsics = cameras["extrinsics"], cameras["intrinsics"]

        context_indices, target_indices = self.view_sampler.sample(scene, extrinsics, intrinsics)

        def views(indices: torch.Tensor) -> dict:
            return {
                "extrinsics": extrinsics[indices],
                "intrinsics": intrinsics[indices],
                "image": self.load_images(scene_dir, cameras["files"], indices),
                "near": torch.full((len(indices),), cameras["near"], dtype=torch.float32),
                "far": torch.full((len(indices),), cameras["far"], dtype=torch.float32),
                "index": indices,
            }

        example = {
            "context": views(context_indices),
            "target": views(target_indices),
            "scene": scene,
        }
        if self.stage == "train" and self.cfg.augment:
            example = apply_augmentation_shim(example)
        if list(example["context"]["image"].shape[2:]) != list(self.cfg.image_shape):
            example = apply_crop_shim(example, tuple(self.cfg.image_shape))
        return example

    def __len__(self) -> int:
        return len(self.scenes)
