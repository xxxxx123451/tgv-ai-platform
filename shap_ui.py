"""ANN 单模型训练后的 SHAP 解释界面。"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from shap_beeswarm import PLOT_LOCK, figure_bytes, plot_shap_beeswarm


def is_single_ann_training(selection_mode, selected):
    """严格限制为侧栏“单模型训练”且模型为 ANN。"""
    return selection_mode == "单模型训练" and list(selected) == ["ANN"]


@st.cache_data(max_entries=3, show_spinner=False)
def _preview(shap_values, feature_values, feature_names, output_name):
    with PLOT_LOCK:
        fig = plot_shap_beeswarm(
            shap_values, feature_values, feature_names,
            output_name=output_name, save_path=None, show=False,
        )
        try:
            return figure_bytes(fig, "png", dpi=150)
        finally:
            plt.close(fig)


def render_ann_shap():
    explanation = st.session_state.get("ann_shap_explanation")
    if explanation is None:
        return
    st.markdown('<div class="section-title">ANN 参数重要性 · SHAP Beeswarm</div>',
                unsafe_allow_html=True)
    st.caption(
        f"仅针对本轮 ANN 单模型训练 · 解释测试集 {len(explanation['feature_values']):,} 个样本 · "
        f"背景样本 {explanation['background_size']:,} 个 · 随机种子 {explanation['seed']}。"
    )
    st.caption(
        "颜色表示该特征自身取值（蓝=低、红=高），横坐标表示该特征相对背景分布对 Area_ratio "
        "预测的推高或压低作用。排序依据为 mean(|SHAP|)，相关性解释不等同于因果关系。"
    )
    st.info(
        "Aspect ratio 自动按 Depth / Wide 计算。SHAP 计算中 Depth 与 Wide 被视为一个联合几何特征组，"
        "二者在每个特征联盟中同时加入或移除；ANN 的训练输入和预测接口仍保持原来的六列不变。"
    )
    st.image(
        _preview(
            explanation["shap_values"], explanation["feature_values"],
            tuple(explanation["feature_names"]), explanation["output_name"],
        ),
        width="stretch",
    )
    st.caption("上方为轻量预览；点击按钮生成 600 dpi PNG 和矢量 PDF。")
    if st.button("🎨 生成 ANN SHAP 投稿图片", key="generate_ann_shap_exports"):
        try:
            with st.spinner("正在生成 600 dpi PNG 和矢量 PDF…"), PLOT_LOCK:
                fig = plot_shap_beeswarm(
                    explanation["shap_values"], explanation["feature_values"],
                    explanation["feature_names"], explanation["output_name"],
                    save_path=None, show=False,
                )
                try:
                    png = figure_bytes(fig, "png", dpi=600)
                    pdf = figure_bytes(fig, "pdf")
                finally:
                    plt.close(fig)
            st.session_state.ann_shap_export = {"png": png, "pdf": pdf}
        except Exception as exc:
            st.error(f"SHAP 图片生成失败：{exc}")
    exported = st.session_state.get("ann_shap_export")
    if exported:
        left, right = st.columns(2)
        left.download_button(
            "⬇️ 下载 SHAP PNG · 600 dpi", exported["png"],
            "ann_shap_beeswarm.png", "image/png", key="download_ann_shap_png",
        )
        right.download_button(
            "⬇️ 下载 SHAP PDF · 矢量图", exported["pdf"],
            "ann_shap_beeswarm.pdf", "application/pdf", key="download_ann_shap_pdf",
        )
    values = pd.DataFrame(
        explanation["shap_values"],
        columns=[f"SHAP_{name}" for name in explanation["feature_names"]],
    )
    features = pd.DataFrame(explanation["feature_values"], columns=explanation["feature_names"])
    table = pd.concat([features, values], axis=1)
    st.download_button(
        "⬇️ 下载 ANN 特征值与 SHAP 值（CSV）",
        table.to_csv(index=False).encode("utf-8-sig"),
        "ann_shap_values.csv", "text/csv", key="download_ann_shap_values",
    )
