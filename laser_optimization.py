"""Residual U-Net and Bayesian optimization framework for TGV laser drilling.

No synthetic observations are produced in this module.  The Bayesian
optimizer must be connected to a real drilling experiment (or be updated with
real measured quality indices) before it can evaluate an objective value.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
import os
from pathlib import Path
import tempfile
from typing import Any, Callable, Iterable, Sequence

import numpy as np

try:  # PyTorch is optional for the original tabular-regression workspace.
    import torch
    from torch import Tensor, nn
except ImportError:  # pragma: no cover - exercised on lightweight installs
    torch = None
    Tensor = Any
    nn = None


QUALITY_METRICS = ("NUI", "SSI", "TI", "MCI", "RI", "ARI")
AUGMENTATION_METHODS = {
    "horizontal_flip": "水平翻转",
    "vertical_flip": "垂直翻转",
    "translation": "垂直平移（Y）",
    "rotation": "小角度旋转（0–5°）",
    "brightness": "亮度调整",
    "gaussian_noise": "高斯噪声",
}
QUALITY_METRIC_DESCRIPTIONS = {
    "NUI": "Normalized Uniformity Index（归一化均匀度指数）：1 表示孔径完全均匀。",
    "SSI": "Sidewall Smoothness Index（侧壁平滑度指数）：1 表示侧壁最平滑。",
    "TI": "Taper Index（锥度指数）：1 表示达到理想锥度。",
    "MCI": "Micro-crack Index（微裂纹指数）：1 表示不存在微裂纹。",
    "RI": "Residue Index（残留指数）：0 表示无残留；该指标越低越好。",
    "ARI": "Aspect Ratio Index（纵横比指数）：1 表示达到期望纵横比。",
}
DEFAULT_CQS_WEIGHTS = (1 / 6,) * 6
LASER_PARAMETER_NAMES = (
    "laser_power_w",
    "pulse_width_ps",
    "repetition_rate_khz",
    "beam_diameter_um",
    "scan_speed_mm_s",
)
LASER_SPACE_SPECS = (
    ("laser_power_w", 5.0, 30.0, "激光功率", "W"),
    ("pulse_width_ps", 0.3, 10.0, "脉冲宽度", "ps"),
    ("repetition_rate_khz", 50.0, 200.0, "重复频率", "kHz"),
    ("beam_diameter_um", 5.0, 50.0, "光束直径", "μm"),
    ("scan_speed_mm_s", 50.0, 200.0, "扫描速度", "mm/s"),
)


if nn is not None:
    class ResidualBlock(nn.Module):
        """Two 3x3 convolutions with a residual shortcut.

        对应文献式 2：X_out = ReLU(X_in + F(X_in, W))。
        当输入、输出通道数不同时，使用 1x1 投影使张量能够相加。
        """

        def __init__(self, in_channels: int, out_channels: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
            self.bn1 = nn.BatchNorm2d(out_channels)
            self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
            self.bn2 = nn.BatchNorm2d(out_channels)
            self.relu = nn.ReLU(inplace=True)
            self.shortcut = (
                nn.Identity()
                if in_channels == out_channels
                else nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                )
            )

        def forward(self, x: Tensor) -> Tensor:
            residual = self.shortcut(x)
            out = self.relu(self.bn1(self.conv1(x)))
            out = self.relu(self.bn2(self.conv2(out)))
            return self.relu(residual + out)  # 对应文献式 2


    class ResidualUNet(nn.Module):
        """Residual U-Net that maps ``[B, 1, H, W]`` to six quality scores.

        ``head_mode='strict'`` exactly uses the requested
        ``Linear(128*224*224, 512)`` head.  It contains roughly 3.29 billion
        parameters (about 12.3 GiB for FP32 weights alone), so the default
        ``'pooled'`` mode inserts global average pooling and is the practical
        training choice.  Both modes preserve the six sigmoid outputs.

        输出顺序：NUI、SSI、TI、MCI、RI、ARI。RI 越低越好，其余指标越高越好。
        """

        def __init__(
            self,
            head_mode: str = "pooled",
            base_channels: int = 64,
            num_outputs: int = 6,
        ) -> None:
            super().__init__()
            if head_mode not in {"strict", "pooled"}:
                raise ValueError("head_mode 必须为 'strict' 或 'pooled'。")
            if num_outputs != 6:
                raise ValueError("TGV 质量评估必须输出 6 个指标。")
            c1, c2, c3, c4, cb = (base_channels * n for n in (1, 2, 4, 8, 16))
            self.head_mode = head_mode
            self.expected_input_size = (224, 224)

            # 编码器：4 个残差块，标准配置通道为 64/128/256/512。
            self.enc1 = ResidualBlock(1, c1)
            self.enc2 = ResidualBlock(c1, c2)
            self.enc3 = ResidualBlock(c2, c3)
            self.enc4 = ResidualBlock(c3, c4)
            self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

            # 瓶颈：标准配置为 1024 通道。
            self.bottleneck = ResidualBlock(c4, cb)

            # 解码器：转置卷积上采样、拼接同尺度编码特征、残差细化。
            self.up4 = nn.ConvTranspose2d(cb, c4, 2, stride=2)
            self.dec4 = ResidualBlock(c4 + c4, c4)
            self.up3 = nn.ConvTranspose2d(c4, c3, 2, stride=2)
            self.dec3 = ResidualBlock(c3 + c3, c3)
            self.up2 = nn.ConvTranspose2d(c3, c2, 2, stride=2)
            self.dec2 = ResidualBlock(c2 + c2, c2)
            self.up1 = nn.ConvTranspose2d(c2, c1, 2, stride=2)
            self.dec1 = ResidualBlock(c1 + c1, c1)

            # 对应文献式 3：C1(x)=ReLU(BN(Conv_3x3(64,256)(x)))。
            self.refine1 = nn.Sequential(
                nn.Conv2d(c1, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True),
            )
            # 对应文献式 4：C2(x)=ReLU(BN(Conv_3x3(256,128)(x)))。
            self.refine2 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True),
            )

            if head_mode == "strict":
                # 对应文献式 5、6；严格复现用户指定的 224x224 全展平预测头。
                self.pool_head = nn.Identity()
                in_features = 128 * 224 * 224
            else:
                # 工程可训练版：避免严格头约 12.3 GiB 的单层权重。
                self.pool_head = nn.AdaptiveAvgPool2d((1, 1))
                in_features = 128
            self.mlp = nn.Sequential(
                nn.Flatten(),
                nn.Linear(in_features, 512),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0.5),
                nn.Linear(512, 6),
                nn.Sigmoid(),
            )

        def forward(self, x: Tensor) -> Tensor:
            if x.ndim != 4 or x.shape[1] != 1:
                raise ValueError("输入必须为 [Batch, 1, H, W] 单通道张量。")
            if self.head_mode == "strict" and tuple(x.shape[-2:]) != self.expected_input_size:
                raise ValueError("strict 模式要求输入空间尺寸严格为 224x224。")
            e1 = self.enc1(x)
            e2 = self.enc2(self.pool(e1))
            e3 = self.enc3(self.pool(e2))
            e4 = self.enc4(self.pool(e3))
            b = self.bottleneck(self.pool(e4))
            d4 = self.dec4(torch.cat((self.up4(b), e4), dim=1))
            d3 = self.dec3(torch.cat((self.up3(d4), e3), dim=1))
            d2 = self.dec2(torch.cat((self.up2(d3), e2), dim=1))
            d1 = self.dec1(torch.cat((self.up1(d2), e1), dim=1))
            features = self.refine2(self.refine1(d1))
            return self.mlp(self.pool_head(features))


else:
    class ResidualBlock:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("Residual U-Net 需要 PyTorch，请先安装 torch。")


    class ResidualUNet:  # type: ignore[no-redef]
        def __init__(self, *_: Any, **__: Any) -> None:
            raise ImportError("Residual U-Net 需要 PyTorch，请先安装 torch。")


def train_one_epoch(
    model: Any,
    dataloader: Iterable[tuple[Any, Any]],
    optimizer: Any,
    device: Any,
    criterion: Any | None = None,
) -> float:
    """Train one epoch with MSE; the caller supplies a real-data DataLoader."""
    if torch is None or nn is None:
        raise ImportError("训练 Residual U-Net 需要 PyTorch。")
    loss_fn = criterion if criterion is not None else nn.MSELoss()
    model.train()
    total_loss = 0.0
    total_samples = 0
    for images, targets in dataloader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True).float()
        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_fn(predictions, targets)  # 六项归一化质量指标的 MSE
        loss.backward()
        optimizer.step()
        batch_size = images.shape[0]
        total_loss += float(loss.detach()) * batch_size
        total_samples += batch_size
    if total_samples == 0:
        raise ValueError("dataloader 为空。")
    return total_loss / total_samples


def prepare_labeled_sem_records(image_files: Sequence[Any], labels_frame: Any) -> list[tuple[str, bytes, list[float]]]:
    """Pair uploaded SEM image files with expert labels by filename.

    The CSV must contain ``filename`` and the six columns in
    ``QUALITY_METRICS``. This function only validates/pairs supplied real data;
    it does not synthesize or augment observations.
    """
    required = ["filename", *QUALITY_METRICS]
    missing = [column for column in required if column not in labels_frame.columns]
    if missing:
        raise ValueError("标签 CSV 缺少列：" + ", ".join(missing))
    label_map: dict[str, list[float]] = {}
    for _, row in labels_frame[required].iterrows():
        key = str(row["filename"]).replace("\\", "/").rsplit("/", 1)[-1].casefold()
        try:
            values = [float(row[name]) for name in QUALITY_METRICS]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key} 的指标必须为 0~1 数值。") from exc
        if not all(np.isfinite(value) and 0.0 <= value <= 1.0 for value in values):
            raise ValueError(f"{key} 的指标必须为 0~1 数值。")
        label_map[key] = values

    records = []
    unmatched = []
    for image_file in image_files:
        filename = image_file.name.replace("\\", "/").rsplit("/", 1)[-1]
        key = filename.casefold()
        if key not in label_map:
            unmatched.append(filename)
            continue
        records.append((filename, image_file.getvalue(), label_map[key]))
    if unmatched:
        raise ValueError("以下图像在标签 CSV 中找不到同名 filename：" + ", ".join(unmatched[:8]))
    if len(records) < 2:
        raise ValueError("至少需要 2 张有标签的 SEM 图像，才能划分训练集与验证集。")
    return records


def augment_sem_image(
    image_bytes: bytes,
    methods: Sequence[str],
    *,
    max_rotation_degrees: float = 5.0,
    max_translation_pixels: int = 10,
    brightness_beta: float = 0.2,
    gaussian_sigma: tuple[float, float] = (0.01, 0.05),
    seed: int = 42,
) -> list[tuple[str, bytes]]:
    """Create one 224x224 transformed image per selected paper-listed method.

    Intensities are normalized to [0, 1]. Brightness follows I'=I+beta,
    beta~Uniform(-brightness_beta, brightness_beta). Gaussian noise follows
    the paper's G_{mu,sigma}(I)=I+N(mu,sigma), with mu=0 and sigma sampled
    from the configured interval (paper: [0.01, 0.05]); values are clipped to
    [0,1] before being encoded back as grayscale PNG.
    """
    from PIL import Image, ImageOps

    unknown = sorted(set(methods).difference(AUGMENTATION_METHODS))
    if unknown:
        raise ValueError("未知增强方法：" + ", ".join(unknown))
    if not 0 <= max_rotation_degrees <= 5:
        raise ValueError("旋转角度上限必须在 0~5 度之间。")
    if max_translation_pixels < 0:
        raise ValueError("平移像素范围不能小于 0。")
    if not 0 <= brightness_beta <= 0.2:
        raise ValueError("亮度 beta 上限必须在 0~0.2 之间。")
    sigma_min, sigma_max = map(float, gaussian_sigma)
    if not (0.01 <= sigma_min <= sigma_max <= 0.05):
        raise ValueError("高斯噪声 sigma 范围必须位于 0.01~0.05。")

    rng = np.random.default_rng(seed)
    with Image.open(BytesIO(image_bytes)) as source:
        base = source.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
    generated: list[tuple[str, bytes]] = []
    for method in methods:
        if method == "horizontal_flip":
            transformed = ImageOps.mirror(base)
        elif method == "vertical_flip":
            transformed = ImageOps.flip(base)
        elif method == "translation":
            # 文献重点描述垂直平移，用于模拟 SEM 定位产生的轻微错位。
            dx = 0
            dy = int(rng.integers(-max_translation_pixels, max_translation_pixels + 1))
            transformed = base.transform(
                base.size,
                Image.Transform.AFFINE,
                (1, 0, -dx, 0, 1, -dy),
                resample=Image.Resampling.BILINEAR,
                fillcolor=0,
            )
        elif method == "rotation":
            angle = float(rng.uniform(0.0, max_rotation_degrees)) if max_rotation_degrees else 0.0
            transformed = base.rotate(
                angle, resample=Image.Resampling.BICUBIC, expand=False, fillcolor=0
            )
        elif method == "brightness":
            beta = float(rng.uniform(-brightness_beta, brightness_beta)) if brightness_beta else 0.0
            pixels = np.asarray(base, dtype=np.float32) / 255.0
            transformed = Image.fromarray(
                np.uint8(np.rint(np.clip(pixels + beta, 0.0, 1.0) * 255.0)), mode="L"
            )
        elif method == "gaussian_noise":
            sigma = float(rng.uniform(sigma_min, sigma_max))
            pixels = np.asarray(base, dtype=np.float32) / 255.0
            # 对应文献高斯噪声公式：G_{mu,sigma}(I)=I+N(mu,sigma), mu=0。
            noise = rng.normal(loc=0.0, scale=sigma, size=pixels.shape).astype(np.float32)
            transformed = Image.fromarray(
                np.uint8(np.rint(np.clip(pixels + noise, 0.0, 1.0) * 255.0)), mode="L"
            )
        else:  # pragma: no cover - guarded by the supported-method check
            continue
        output = BytesIO()
        transformed.save(output, format="PNG")
        generated.append((method, output.getvalue()))
    return generated


def augment_labeled_sem_records(
    records: Sequence[tuple[str, bytes, Sequence[float]]],
    methods: Sequence[str],
    **augmentation_options: Any,
) -> list[tuple[str, bytes, list[float]]]:
    """Return original labeled samples plus transformed copies with same labels."""
    expanded = [(name, image, list(map(float, labels))) for name, image, labels in records]
    for index, (name, image, labels) in enumerate(records):
        stem = Path(name).stem
        for method, transformed in augment_sem_image(
            image, methods, seed=int(augmentation_options.get("seed", 42)) + index,
            **{key: value for key, value in augmentation_options.items() if key != "seed"},
        ):
            expanded.append((f"{stem}__aug_{method}.png", transformed, list(map(float, labels))))
    return expanded


def train_quality_model(
    records: Sequence[tuple[str, bytes, Sequence[float]]],
    epochs: int = 5,
    batch_size: int = 1,
    learning_rate: float = 1e-4,
    base_channels: int = 16,
    progress_callback: Callable[[int, float, float | None], None] | None = None,
) -> tuple[Any, list[dict[str, float | None]]]:
    """Train from uploaded, labeled SEM images and return model plus losses."""
    if torch is None or nn is None:
        raise ImportError("训练 Residual U-Net 需要 PyTorch。")
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Subset

    class UploadedSemDataset(Dataset):
        def __init__(self, samples: Sequence[tuple[str, bytes, Sequence[float]]]) -> None:
            self.samples = samples

        def __len__(self) -> int:
            return len(self.samples)

        def __getitem__(self, index: int) -> tuple[Any, Any]:
            _, image_bytes, target = self.samples[index]
            with Image.open(BytesIO(image_bytes)) as image:
                image = image.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
                array = np.asarray(image, dtype=np.float32).copy() / 255.0
            tensor = torch.from_numpy(array).unsqueeze(0)
            target_tensor = torch.tensor(target, dtype=torch.float32)
            return tensor, target_tensor

    if len(records) < 2:
        raise ValueError("至少需要 2 张有标签图像。")
    dataset = UploadedSemDataset(records)
    # Same-source augmented images must stay in one split; otherwise near-
    # duplicates can leak from training into validation and inflate metrics.
    group_indices: dict[str, list[int]] = {}
    for index, (filename, _, _) in enumerate(records):
        group = Path(filename).stem.split("__aug_", 1)[0].casefold()
        group_indices.setdefault(group, []).append(index)
    groups = sorted(group_indices)
    if len(groups) < 2:
        raise ValueError("至少需要 2 张不同的原始 SEM 图像，才能避免增强副本造成验证泄漏。")
    rng = np.random.default_rng(42)
    shuffled_groups = list(rng.permutation(groups))
    validation_group_count = max(1, int(round(len(groups) * 0.2)))
    validation_groups = set(shuffled_groups[:validation_group_count])
    train_indices = [i for group, indices in group_indices.items() if group not in validation_groups for i in indices]
    validation_indices = [i for group, indices in group_indices.items() if group in validation_groups for i in indices]
    train_set = Subset(dataset, train_indices)
    validation_set = Subset(dataset, validation_indices)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    validation_loader = DataLoader(validation_set, batch_size=batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ResidualUNet(head_mode="pooled", base_channels=base_channels).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    history: list[dict[str, float | None]] = []

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, device, criterion)
        model.eval()
        total, count = 0.0, 0
        with torch.no_grad():
            for images, targets in validation_loader:
                images, targets = images.to(device), targets.to(device)
                loss = criterion(model(images), targets)
                total += float(loss) * len(images)
                count += len(images)
        validation_loss = total / count if count else None
        item = {"epoch": float(epoch + 1), "train_mse": train_loss, "validation_mse": validation_loss}
        history.append(item)
        if progress_callback is not None:
            progress_callback(epoch + 1, train_loss, validation_loss)
    model.eval()
    return model, history


def predict_sem_image(model: Any, image_bytes: bytes) -> np.ndarray:
    """Predict six normalized quality indices for one uploaded SEM image."""
    if torch is None:
        raise ImportError("图像预测需要 PyTorch。")
    from PIL import Image

    with Image.open(BytesIO(image_bytes)) as image:
        image = image.convert("L").resize((224, 224), Image.Resampling.BILINEAR)
        array = np.asarray(image, dtype=np.float32).copy() / 255.0
    parameter = next(model.parameters())
    tensor = torch.from_numpy(array).unsqueeze(0).unsqueeze(0).to(parameter.device)
    model.eval()
    with torch.no_grad():
        scores = model(tensor).squeeze(0).detach().cpu().numpy()
    return scores


def calculate_cqs(
    scores: Sequence[float] | Any,
    weights: Sequence[float] = DEFAULT_CQS_WEIGHTS,
) -> float | Any:
    """Calculate the Composite Quality Score (paper equation 1).

    CQS = w1*NUI + w2*SSI + w3*TI + w4*MCI - w5*RI + w6*ARI.
    """
    if len(scores) != 6 or len(weights) != 6:
        raise ValueError("scores 和 weights 都必须包含 6 个值。")
    weight_array = np.asarray(weights, dtype=float)
    if np.any(weight_array < 0) or not np.isclose(weight_array.sum(), 1.0, atol=1e-8):
        raise ValueError("权重必须非负，且权重之和必须为 1。")
    if torch is not None and isinstance(scores, torch.Tensor):
        if scores.shape[-1] != 6:
            raise ValueError("scores 最后一维必须包含 6 个指标。")
        w = torch.as_tensor(weight_array, dtype=scores.dtype, device=scores.device)
        signs = torch.tensor([1, 1, 1, 1, -1, 1], dtype=scores.dtype, device=scores.device)
        return (scores * w * signs).sum(dim=-1)
    values = np.asarray(scores, dtype=float)
    if values.shape != (6,) or np.any(~np.isfinite(values)):
        raise ValueError("scores 必须是 6 个有限数值。")
    if np.any((values < 0) | (values > 1)):
        raise ValueError("六个质量指标必须归一化到 0~1。")
    return float(np.dot(values, weight_array * np.array([1, 1, 1, 1, -1, 1])))


def get_search_space() -> list[Any]:
    """Return the five paper-table-1 dimensions as ``skopt.space.Real``."""
    try:
        from skopt.space import Real
    except ImportError as exc:
        raise ImportError("贝叶斯优化需要 scikit-optimize，请安装 scikit-optimize。") from exc
    return [Real(low, high, name=name) for name, low, high, _, _ in LASER_SPACE_SPECS]


def objective_function(params: Sequence[float]) -> float:
    """Placeholder for one *real* drilling experiment.

    Connect the five parameters to the laser, acquire the SEM image, infer the
    six scores with ``ResidualUNet``, calculate CQS, then return ``1 - CQS``.
    """
    if len(params) != 5:
        raise ValueError("params 必须依次包含 5 个激光参数。")
    raise NotImplementedError(
        "请在 objective_function 中连接真实打孔实验和 SEM 质量评估；框架不会生成模拟结果。"
    )


def run_bayesian_optimization(
    objective: Callable[[Sequence[float]], float] = objective_function,
    n_calls: int = 50,
    kappa: float = 2.0,
    random_state: int = 42,
) -> Any:
    """Run GP Bayesian optimization for 50 evaluations.

    ``gp_minimize`` minimizes ``1-CQS``.  Therefore scikit-optimize's LCB on
    the loss is mathematically equivalent to maximizing the paper's UCB on
    CQS: ``mu_CQS + kappa*sigma_CQS`` (paper equation 8).
    """
    try:
        from skopt import gp_minimize
    except ImportError as exc:
        raise ImportError("贝叶斯优化需要 scikit-optimize。") from exc
    if n_calls < 10:
        raise ValueError("gp_minimize 至少需要 10 次调用（默认 50）。")
    return gp_minimize(
        objective,
        dimensions=get_search_space(),
        n_calls=n_calls,
        acq_func="LCB",
        acq_func_kwargs={"kappa": float(kappa)},
        random_state=random_state,
    )


@dataclass
class IterativeBayesianOptimizer:
    """Ask/tell adapter for sequential real experiments in the web UI."""

    kappa: float = 2.0
    random_state: int = 42
    n_initial_points: int = 5
    history_path: str | Path | None = None

    def __post_init__(self) -> None:
        try:
            from skopt import Optimizer
        except ImportError as exc:
            raise ImportError("迭代优化需要 scikit-optimize。") from exc
        self._optimizer = Optimizer(
            get_search_space(),
            base_estimator="GP",
            acq_func="LCB",
            acq_func_kwargs={"kappa": float(self.kappa)},
            n_initial_points=self.n_initial_points,
            random_state=self.random_state,
        )
        default_path = Path(__file__).resolve().parent / "data" / "laser_bo_history.csv"
        self.history_path = Path(self.history_path or os.environ.get("TGV_LASER_BO_HISTORY", default_path))
        self.history: list[dict[str, Any]] = self._load_persisted_history()
        # Keep the archive across restarts, but start the active surrogate empty.
        self.session_observations: list[dict[str, Any]] = []

    def _load_persisted_history(self) -> list[dict[str, Any]]:
        if not self.history_path.exists():
            return []
        columns = [
            "recorded_at", *LASER_PARAMETER_NAMES, *QUALITY_METRICS, "CQS",
            *(f"weight_{name}" for name in QUALITY_METRICS),
        ]
        rows: list[dict[str, Any]] = []
        with self.history_path.open("r", newline="", encoding="utf-8-sig") as stream:
            reader = csv.DictReader(stream)
            missing = set(columns).difference(reader.fieldnames or [])
            if missing:
                raise ValueError("历史数据文件缺少字段：" + ", ".join(sorted(missing)))
            for line_number, raw in enumerate(reader, start=2):
                try:
                    row: dict[str, Any] = {"recorded_at": raw["recorded_at"]}
                    row.update({name: float(raw[name]) for name in LASER_PARAMETER_NAMES})
                    row.update({name: float(raw[name]) for name in QUALITY_METRICS})
                    row["CQS"] = float(raw["CQS"])
                    row.update({f"weight_{name}": float(raw[f"weight_{name}"]) for name in QUALITY_METRICS})
                    point = [row[name] for name in LASER_PARAMETER_NAMES]
                    if not all(np.isfinite(value) for value in point) or not np.isfinite(row["CQS"]):
                        raise ValueError("包含非有限数值")
                    rows.append(row)
                except (TypeError, ValueError, KeyError) as exc:
                    raise ValueError(f"历史数据文件第 {line_number} 行无效：{exc}") from exc
        return rows

    def _save_persisted_history(self) -> None:
        self.history_path.parent.mkdir(parents=True, exist_ok=True)
        columns = [
            "recorded_at", *LASER_PARAMETER_NAMES, *QUALITY_METRICS, "CQS",
            *(f"weight_{name}" for name in QUALITY_METRICS),
        ]
        temp_name: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", newline="", encoding="utf-8-sig", dir=self.history_path.parent,
                prefix=self.history_path.name + ".", suffix=".tmp", delete=False,
            ) as stream:
                temp_name = stream.name
                writer = csv.DictWriter(stream, fieldnames=columns)
                writer.writeheader()
                writer.writerows(self.history)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temp_name, self.history_path)
        finally:
            if temp_name and os.path.exists(temp_name):
                os.unlink(temp_name)

    def ask(self) -> dict[str, float]:
        return dict(zip(LASER_PARAMETER_NAMES, map(float, self._optimizer.ask())))

    def tell(
        self,
        params: Sequence[float],
        scores: Sequence[float],
        weights: Sequence[float] = DEFAULT_CQS_WEIGHTS,
    ) -> dict[str, Any]:
        if len(params) != 5:
            raise ValueError("必须提供 5 个激光参数。")
        cqs = float(calculate_cqs(scores, weights))
        loss = 1.0 - cqs
        point = list(map(float, params))
        self._optimizer.tell(point, loss)
        row = {
            "recorded_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            **dict(zip(LASER_PARAMETER_NAMES, point)),
            **dict(zip(QUALITY_METRICS, map(float, scores))),
            "CQS": cqs,
            **{f"weight_{name}": float(weight) for name, weight in zip(QUALITY_METRICS, weights)},
        }
        self.history.append(row)
        self.session_observations.append(row)
        self._save_persisted_history()
        return row

