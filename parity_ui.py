"""训练完成后的测试集可视化与投稿图片下载。"""
import matplotlib
matplotlib.use("Agg")  # 云端无显示器，不能弹出桌面绘图窗口。
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

from parity_plot import MODEL_ORDER, PLOT_LOCK, figure_bytes, plot_parity


@st.cache_data(max_entries=4, show_spinner=False)
def preview_bytes(y_true, predictions):
    # 低分辨率预览与投稿导出分离，避免每次填写输入参数都生成大图。
    with PLOT_LOCK:
        fig = plot_parity(y_true, predictions, save_path=None)
        try:
            return figure_bytes(fig, "png", dpi=140)
        finally:
            plt.close(fig)


def render_parity():
    evaluation = st.session_state.get("parity_evaluation")
    if evaluation is None:
        return
    st.markdown('<div class="section-title">真实值 vs 预测值 · Parity Plot</div>', unsafe_allow_html=True)
    st.caption(f"独立测试集：{len(evaluation['y_true']):,} 个样本 · "
               f"本轮训练随机种子 {evaluation['seed']} · 测试比例 {evaluation['test_size']:.0%}。"
               "所有模型使用相同的测试样本与顺序；不使用训练集拟合值，不截断超出 0～1 的预测。")
    st.caption("参考文献 Figure 10 的视觉风格；本图指标仅对应 Area_ratio，"
               "不是论文中五个输出目标的平均指标。RMAE 为沿用的图注标签，实际计算 RMSE。")
    names = [name for name in MODEL_ORDER if name in evaluation["predictions"]]
    selection = st.selectbox("选择对比图", ["全部已训练模型"] + names, key="parity_selection")
    predictions = {name: evaluation["predictions"][name] for name in names
                   if selection == "全部已训练模型" or name == selection}
    if len(predictions) < 6 and selection == "全部已训练模型":
        st.caption("仅绘制本轮成功训练的模型；训练六种模型后显示完整 2×3 对比图。")
    st.image(preview_bytes(evaluation["y_true"], predictions), use_container_width=True)
    st.caption("上方为轻量预览。点击下方按钮生成 600 dpi PNG 与矢量 PDF，生成后即可下载。")
    export_key = tuple(predictions)
    if st.button("🖼️ 生成投稿图片（PNG + PDF）", key="generate_parity_exports"):
        try:
            with st.spinner("正在生成 600 dpi PNG 和矢量 PDF…"), PLOT_LOCK:
                fig = plot_parity(evaluation["y_true"], predictions, save_path=None)
                try:
                    png = figure_bytes(fig, "png", dpi=600)
                    pdf = figure_bytes(fig, "pdf")
                finally:
                    plt.close(fig)
            # 只保留当前选项的导出，避免大型图片在云端持续累积。
            st.session_state.parity_export = {"key": export_key, "png": png, "pdf": pdf}
        except Exception as exc:
            st.error(f"图片生成失败：{exc}")
    exported = st.session_state.get("parity_export")
    if exported and exported["key"] == export_key:
        stem = "parity_plots" if selection == "全部已训练模型" else f"parity_{selection}"
        left, right = st.columns(2)
        left.download_button("⬇️ 下载 PNG · 600 dpi", exported["png"], f"{stem}.png",
                             "image/png", key="download_parity_png")
        right.download_button("⬇️ 下载 PDF · 矢量图", exported["pdf"], f"{stem}.pdf",
                              "application/pdf", key="download_parity_pdf")
    # 提供原始测试预测，方便核对指标、重新绘图及论文数据归档。
    table = pd.DataFrame({"source_index": evaluation["source_index"],
                          "y_true": evaluation["y_true"], **evaluation["predictions"]})
    st.download_button("⬇️ 下载测试集真实值与预测值（CSV）",
                       table.to_csv(index=False).encode("utf-8-sig"),
                       "test_predictions.csv", "text/csv", key="download_parity_data")

