"""论文风格真实值/预测值对比图；可导入网页，也可独立运行。

命令行：python parity_plot.py --csv test_predictions.csv --output parity_plots.png
CSV 必须包含 y_true 以及一个或多个 ANN/XGBOOST/RF/GPR/SVR/PR 列。
仅使用真实测试集输出，不生成模拟数据，不裁剪原始数值。
"""
import argparse
import io
from pathlib import Path
from threading import RLock

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

MODEL_ORDER = ("ANN", "XGBOOST", "RF", "GPR", "SVR", "PR")
# Streamlit 多用户共享 matplotlib；绘制、导出和关闭必须处于同一锁内。
PLOT_LOCK = RLock()
FONT_STYLE = {"font.family": "DejaVu Sans", "font.size": 12,
              "pdf.fonttype": 42, "ps.fonttype": 42,
              "axes.facecolor": "white", "figure.facecolor": "white",
              "text.color": "black", "axes.labelcolor": "black",
              "xtick.color": "black", "ytick.color": "black"}


def plot_parity(y_true, predictions, save_path="parity_plots.png"):
    """返回 Figure；save_path 非 None 时保存 600 dpi PNG 和同名矢量 PDF。

    六模型固定 2×3；单模型为单图，其他数量保留 2×3 槽位并隐藏空轴。
    y_true 与每个预测数组必须同序、一维、有限，且至少有两个样本。
    面积比原则上为 0~1，但不截断异常/外推值，以免掩盖误差。
    RMSE 为均方根误差，MAE 为平均绝对误差。
    调用方负责 plt.close(fig)；网页不调用阻塞式 plt.show()。
    """
    truth = np.asarray(y_true, dtype=float)
    if truth.ndim != 1 or truth.size < 2 or not np.isfinite(truth).all():
        raise ValueError("y_true 必须是至少含两个有限数值的一维数组。")
    if not isinstance(predictions, dict) or not predictions:
        raise ValueError("predictions 必须是非空模型预测字典。")
    if set(predictions) - set(MODEL_ORDER):
        raise ValueError("支持的模型名为：" + ", ".join(MODEL_ORDER))
    values = {}
    for name in MODEL_ORDER:
        if name not in predictions:
            continue
        pred = np.asarray(predictions[name], dtype=float)
        if pred.shape != truth.shape or not np.isfinite(pred).all():
            raise ValueError(f"{name} 的预测必须与 y_true 同形状且全部有限。")
        values[name] = pred

    with PLOT_LOCK, matplotlib.rc_context(FONT_STYLE):
        if len(values) == 1:
            fig, ax = plt.subplots(figsize=(5.3, 5.0))
            axes = np.array([ax])
        else:
            fig, axes = plt.subplots(2, 3, figsize=(14, 9))
            axes = axes.ravel()
        fig.subplots_adjust(hspace=0.35, wspace=0.30)
        for index, (name, pred) in enumerate(values.items()):
            ax = axes[index]
            mae = mean_absolute_error(truth, pred)
            rmse = np.sqrt(mean_squared_error(truth, pred))
            r2 = r2_score(truth, pred)
            min_val = min(float(truth.min()), float(pred.min()))
            max_val = max(float(truth.max()), float(pred.max()))
            span = max_val - min_val
            # 常量数据仍需非零坐标跨度。
            padding = 0.05 * (span if span > 0 else max(abs(min_val), 1.0))
            bounds = (min_val - padding, max_val + padding)
            ax.scatter(truth, pred, s=18, c="royalblue", alpha=0.55, edgecolors="none")
            ax.plot([min_val, max_val], [min_val, max_val], color="red", linewidth=1.6)
            ax.set_xlim(bounds)
            ax.set_ylim(bounds)
            ax.set_aspect("equal", adjustable="box")
            ax.set_xlabel("True", fontsize=14)
            ax.set_ylabel("Prediction value", fontsize=14)
            ax.tick_params(axis="both", labelsize=12)
            ax.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_linewidth(1)
                spine.set_color("black")
            ax.text(-0.16, 1.05, f"({chr(97 + index)})", transform=ax.transAxes,
                    fontsize=14, fontweight="bold", va="bottom", clip_on=False)
            ax.text(0.04, 0.96, "Area rate", transform=ax.transAxes,
                    color="red", fontsize=14, fontweight="bold", va="top", ha="left")
            for y, label in zip((0.86, 0.78, 0.70),
                                (f"MAE = {mae:.4f}", f"RMSE = {rmse:.4f}", f"R² = {r2:.4f}")):
                ax.text(0.04, y, label, transform=ax.transAxes,
                        color="black", fontsize=12, va="top", ha="left")
            ax.text(0.95, 0.06, name, transform=ax.transAxes, fontsize=22,
                    fontweight="bold", ha="right", va="bottom", color="black")
        for ax in axes[len(values):]:
            ax.set_visible(False)
        if save_path is not None:
            path = Path(save_path)
            if path.suffix.lower() != ".png":
                plt.close(fig)
                raise ValueError("save_path 必须以 .png 结尾；同时自动保存同名 .pdf。")
            try:
                fig.savefig(path, dpi=600, bbox_inches="tight", facecolor="white")
                fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", facecolor="white")
            except Exception:
                plt.close(fig)
                raise
        return fig


def figure_bytes(fig, format="png", dpi=600):
    """导出到内存，供网页下载；PDF 中的散点和文字保持矢量。"""
    if format not in ("png", "pdf"):
        raise ValueError("仅支持 png 或 pdf。")
    with PLOT_LOCK, matplotlib.rc_context(FONT_STYLE):
        buffer = io.BytesIO()
        fig.savefig(buffer, format=format, dpi=dpi, bbox_inches="tight", facecolor="white")
        return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser(description="从真实测试集预测 CSV 绘制论文风格 Parity Plot")
    parser.add_argument("--csv", required=True, help="含 y_true 和模型预测列的 CSV")
    parser.add_argument("--output", default="parity_plots.png")
    args = parser.parse_args()
    data = np.genfromtxt(args.csv, delimiter=",", names=True, encoding="utf-8-sig")
    data = np.atleast_1d(data)
    if not data.dtype.names or "y_true" not in data.dtype.names:
        parser.error("CSV 缺少 y_true 列。")
    predictions = {name: data[name] for name in MODEL_ORDER if name in data.dtype.names}
    fig = plot_parity(data["y_true"], predictions, args.output)
    print(f"已保存：{args.output} 和 {Path(args.output).with_suffix('.pdf')}")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
