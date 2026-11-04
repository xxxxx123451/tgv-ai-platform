"""论文风格 SHAP Summary Beeswarm Plot 与小特征集精确 SHAP 计算。

真实使用时通常先用 SHAP 官方库获得数值：

    import shap
    explainer = shap.TreeExplainer(model)  # 或 KernelExplainer / DeepExplainer
    shap_values = explainer.shap_values(X)

本文件的 ``plot_shap_beeswarm`` 可直接绘制上述结果。为使网页中的
scikit-learn ANN 不依赖特定解释器，另提供 ``exact_interventional_shap``：
它枚举所有特征联盟，并相对背景样本计算精确的介入式 Shapley 值。
"""
from __future__ import annotations

import io
from pathlib import Path
from threading import RLock

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import numpy as np
import pandas as pd
from scipy.special import factorial


PLOT_LOCK = RLock()
PLOT_STYLE = {
    "font.family": "DejaVu Sans",
    "font.size": 12,
    "axes.linewidth": 1.0,
    "savefig.dpi": 600,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "text.color": "black",
    "axes.labelcolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "axes.facecolor": "white",
    "figure.facecolor": "white",
}
matplotlib.rcParams.update(PLOT_STYLE)


def _beeswarm_offsets(values, max_half_height=0.42, n_bins=60):
    """按横坐标分箱，以 0,+1,-1,+2,-2... 形成简化蜂群堆叠。"""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("values 必须是一维有限数值数组。")
    if values.size == 0:
        return np.array([], dtype=float)
    if max_half_height <= 0 or n_bins < 1:
        raise ValueError("max_half_height 必须大于 0，n_bins 必须至少为 1。")
    vmin, vmax = float(values.min()), float(values.max())
    if vmax == vmin:
        return np.zeros(values.size, dtype=float)

    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    bins = np.floor((sorted_values - vmin) / (vmax - vmin) * n_bins).astype(int)
    bins = np.clip(bins, 0, n_bins - 1)
    counts = np.bincount(bins, minlength=n_bins)
    scale = max(float(counts.max()) / 2.0, 1.0)
    sorted_offsets = np.zeros(values.size, dtype=float)
    for bin_id in np.flatnonzero(counts):
        positions = np.flatnonzero(bins == bin_id)
        for i, position in enumerate(positions):
            if i == 0:
                level = 0.0
            else:
                level = float((i + 1) // 2) * (1.0 if i % 2 else -1.0)
            sorted_offsets[position] = level / scale * max_half_height
    sorted_offsets = np.clip(sorted_offsets, -max_half_height, max_half_height)
    offsets = np.empty_like(sorted_offsets)
    offsets[order] = sorted_offsets
    return offsets


def _validate_plot_inputs(shap_values, feature_values, feature_names):
    shap_array = np.asarray(shap_values, dtype=float)
    feature_array = np.asarray(feature_values, dtype=float)
    if shap_array.ndim != 2 or not shap_array.size or not np.isfinite(shap_array).all():
        raise ValueError("shap_values 必须是非空且全部有限的二维数组。")
    if feature_array.shape != shap_array.shape or not np.isfinite(feature_array).all():
        raise ValueError("feature_values 必须与 shap_values 同形状且全部有限。")
    names = list(feature_names)
    if len(names) != shap_array.shape[1] or not all(isinstance(n, str) and n for n in names):
        raise ValueError("feature_names 必须与特征列数一致，且均为非空字符串。")
    return shap_array, feature_array, names


def plot_shap_beeswarm(
    shap_values,
    feature_values,
    feature_names,
    output_name="Output-1",
    save_path="shap_beeswarm.png",
    *,
    show=True,
):
    """按 mean(|SHAP|) 排序并返回 Figure，同时可保存 600 dpi PNG/PDF。"""
    shap_array, feature_array, names = _validate_plot_inputs(
        shap_values, feature_values, feature_names
    )
    importance_order = np.argsort(np.mean(np.abs(shap_array), axis=0))[::-1]
    n_features = len(names)

    with PLOT_LOCK, matplotlib.rc_context(PLOT_STYLE):
        fig, ax = plt.subplots(figsize=(7.5, 5.5))
        fig.subplots_adjust(left=0.28, right=0.86, bottom=0.17, top=0.95)
        for rank, feature_index in enumerate(importance_order):
            y_center = n_features - 1 - rank  # 重要性最高者位于顶部（最大 y）。
            x_j = shap_array[:, feature_index]
            c_j = feature_array[:, feature_index]
            c_norm = (c_j - c_j.min()) / (c_j.max() - c_j.min() + 1e-12)
            offsets = _beeswarm_offsets(x_j)
            ax.scatter(
                x_j, y_center + offsets, c=c_norm, cmap="coolwarm",
                s=10, alpha=0.85, edgecolors="none", zorder=2,
            )

        x_min, x_max = float(shap_array.min()), float(shap_array.max())
        span = x_max - x_min
        padding = 0.05 * (span if span > 0 else max(abs(x_min), 1.0))
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(-0.6, n_features - 0.4)
        ax.axvline(0, color="gray", linewidth=0.8, alpha=0.6, zorder=0)
        # 刻度按底到顶排列；顶部仍是排序后的首个（最重要）特征。
        bottom_to_top = [names[i] for i in importance_order[::-1]]
        ax.set_yticks(np.arange(n_features), labels=bottom_to_top)
        ax.tick_params(axis="y", length=0, pad=12)
        for label in ax.get_yticklabels():
            label.set_fontsize(14)
            label.set_fontweight("bold")
            label.set_color("black")
            label.set_horizontalalignment("right")
        ax.set_xlabel(
            f"SHAP value (Impact on {output_name})", fontsize=14, fontweight="bold"
        )
        ax.tick_params(axis="x", labelsize=12, direction="out")
        ax.grid(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

        mappable = ScalarMappable(cmap="coolwarm", norm=Normalize(vmin=0, vmax=1))
        mappable.set_array([])
        cbar = fig.colorbar(mappable, ax=ax, fraction=0.04, pad=0.03)
        cbar.set_ticks([])
        cbar.outline.set_visible(False)
        cbar.ax.text(0.5, 1.02, "High", transform=cbar.ax.transAxes,
                     fontsize=12, fontweight="bold", color="black", ha="center", va="bottom")
        cbar.ax.text(0.5, -0.02, "Low", transform=cbar.ax.transAxes,
                     fontsize=12, fontweight="bold", color="black", ha="center", va="top")
        cbar.set_label("Feature value (Low → High)", fontsize=12, fontweight="bold",
                       rotation=270, labelpad=18, color="black")

        if save_path is not None:
            path = Path(save_path)
            if path.suffix.lower() != ".png":
                plt.close(fig)
                raise ValueError("save_path 必须以 .png 结尾。")
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
            fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            print(f"已保存：{path} 和 {path.with_suffix('.pdf')}")
        if show:
            plt.show()
        return fig


def figure_bytes(fig, format="png", dpi=600):
    """把已有图导出到内存，供 Streamlit 下载。"""
    if format not in {"png", "pdf"}:
        raise ValueError("仅支持 png 或 pdf。")
    with PLOT_LOCK, matplotlib.rc_context(PLOT_STYLE):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", facecolor="white")
        return buffer.getvalue()


def _predict(model, values, feature_names):
    model_input = (pd.DataFrame(values, columns=feature_names)
                   if getattr(model, "feature_names_in_", None) is not None else values)
    result = np.asarray(model.predict(model_input), dtype=float).reshape(-1)
    if result.size != len(values) or not np.isfinite(result).all():
        raise ValueError("模型 predict 必须为每行返回一个有限预测值。")
    return result


def exact_interventional_shap(
    model,
    feature_values,
    background_values,
    *,
    feature_names=None,
    max_samples=500,
    max_background=16,
    random_state=42,
):
    """枚举全部联盟，计算相对背景分布的精确介入式 Shapley 值。

    该算法随特征数呈 2^M 增长，适合本平台仅有 6 个输入特征的 ANN；
    不应直接用于几十或上百特征的模型。
    """
    x = np.asarray(feature_values, dtype=float)
    background = np.asarray(background_values, dtype=float)
    if x.ndim != 2 or background.ndim != 2 or x.shape[1] != background.shape[1]:
        raise ValueError("feature_values 与 background_values 必须是列数相同的二维数组。")
    if x.shape[0] == 0 or background.shape[0] == 0:
        raise ValueError("待解释数据和背景数据不能为空。")
    if not np.isfinite(x).all() or not np.isfinite(background).all():
        raise ValueError("待解释数据和背景数据必须全部有限。")
    m = x.shape[1]
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(m)]
    if len(names) != m:
        raise ValueError("feature_names 长度不匹配。")
    rng = np.random.default_rng(random_state)
    if max_samples is not None and len(x) > max_samples:
        selected = np.sort(rng.choice(len(x), max_samples, replace=False))
        x = x[selected]
    if max_background is not None and len(background) > max_background:
        selected = np.sort(rng.choice(len(background), max_background, replace=False))
        background = background[selected]

    mask_ids = np.arange(2 ** m, dtype=int)
    masks = ((mask_ids[:, None] >> np.arange(m)) & 1).astype(bool)
    shap_values = np.zeros_like(x, dtype=float)
    expected_value = float(_predict(model, background, names).mean())
    denominator = float(factorial(m, exact=True))
    for baseline in background:
        mixed = np.where(masks[None, :, :], x[:, None, :], baseline[None, None, :])
        coalition_predictions = _predict(model, mixed.reshape(-1, m), names).reshape(len(x), -1)
        for j in range(m):
            absent = (mask_ids & (1 << j)) == 0
            base_ids = mask_ids[absent]
            sizes = masks[absent].sum(axis=1)
            weights = np.array([
                factorial(int(size), exact=True)
                * factorial(m - int(size) - 1, exact=True) / denominator
                for size in sizes
            ])
            differences = (
                coalition_predictions[:, base_ids | (1 << j)]
                - coalition_predictions[:, base_ids]
            )
            shap_values[:, j] += differences @ weights
    shap_values /= len(background)
    return shap_values, x, expected_value


def exact_grouped_interventional_shap(
    model,
    feature_values,
    background_values,
    *,
    feature_groups,
    feature_names=None,
    max_samples=500,
    max_background=16,
    random_state=42,
):
    """计算分组介入式 Shapley 值，组内原始变量始终同时加入或移除。

    例如 ``((0, 1), (2,), ...)`` 会把 Depth/Wide 视为一个联合几何玩家，
    从而得到保持局部准确性（可加性）的几何组贡献。
    """
    x = np.asarray(feature_values, dtype=float)
    background = np.asarray(background_values, dtype=float)
    if x.ndim != 2 or background.ndim != 2 or x.shape[1] != background.shape[1]:
        raise ValueError("feature_values 与 background_values 必须是列数相同的二维数组。")
    if x.shape[0] == 0 or background.shape[0] == 0:
        raise ValueError("待解释数据和背景数据不能为空。")
    if not np.isfinite(x).all() or not np.isfinite(background).all():
        raise ValueError("待解释数据和背景数据必须全部有限。")
    m = x.shape[1]
    names = list(feature_names) if feature_names is not None else [f"x{i}" for i in range(m)]
    if len(names) != m:
        raise ValueError("feature_names 长度不匹配。")
    groups = tuple(tuple(int(index) for index in group) for group in feature_groups)
    flattened = [index for group in groups for index in group]
    if not groups or any(not group for group in groups) or sorted(flattened) != list(range(m)):
        raise ValueError("feature_groups 必须无遗漏、无重复地覆盖全部原始特征。")

    rng = np.random.default_rng(random_state)
    if max_samples is not None and len(x) > max_samples:
        selected = np.sort(rng.choice(len(x), max_samples, replace=False))
        x = x[selected]
    if max_background is not None and len(background) > max_background:
        selected = np.sort(rng.choice(len(background), max_background, replace=False))
        background = background[selected]

    group_count = len(groups)
    coalition_ids = np.arange(2 ** group_count, dtype=int)
    group_masks = ((coalition_ids[:, None] >> np.arange(group_count)) & 1).astype(bool)
    feature_masks = np.zeros((len(coalition_ids), m), dtype=bool)
    for group_index, original_indices in enumerate(groups):
        feature_masks[:, original_indices] = group_masks[:, [group_index]]
    grouped_values = np.zeros((len(x), group_count), dtype=float)
    expected_value = float(_predict(model, background, names).mean())
    denominator = float(factorial(group_count, exact=True))
    for baseline in background:
        mixed = np.where(
            feature_masks[None, :, :], x[:, None, :], baseline[None, None, :]
        )
        coalition_predictions = _predict(
            model, mixed.reshape(-1, m), names
        ).reshape(len(x), -1)
        for group_index in range(group_count):
            absent = (coalition_ids & (1 << group_index)) == 0
            base_ids = coalition_ids[absent]
            sizes = group_masks[absent].sum(axis=1)
            weights = np.array([
                factorial(int(size), exact=True)
                * factorial(group_count - int(size) - 1, exact=True) / denominator
                for size in sizes
            ])
            differences = (
                coalition_predictions[:, base_ids | (1 << group_index)]
                - coalition_predictions[:, base_ids]
            )
            grouped_values[:, group_index] += differences @ weights
    grouped_values /= len(background)
    return grouped_values, x, expected_value


if __name__ == "__main__":
    # 仅用于快速查看图形风格；网页训练解释绝不使用这些模拟值。
    rng = np.random.default_rng(42)
    n_samples = 3000
    feature_names = ["Aspect ratio", "c_SUP", "i_0", "c_ACC", "c_LEVE"]
    feature_values = rng.uniform(0, 1, size=(n_samples, len(feature_names)))
    noise = rng.normal(0, 1, size=feature_values.shape)
    shap_values = np.column_stack([
        -0.36 * (feature_values[:, 0] - 0.5) + 0.018 * noise[:, 0],
         0.28 * (feature_values[:, 1] - 0.5) + 0.020 * noise[:, 1],
        -0.22 * (feature_values[:, 2] - 0.5) + 0.022 * noise[:, 2],
         0.025 * (feature_values[:, 3] - 0.5) + 0.008 * noise[:, 3],
         0.008 * (feature_values[:, 4] - 0.5) + 0.004 * noise[:, 4],
    ])
    figure = plot_shap_beeswarm(
        shap_values, feature_values, feature_names,
        output_name="Output-1", save_path="shap_beeswarm.png",
    )
    plt.close(figure)
