"""Dopamine-themed Streamlit front end for the TGV regression toolkit."""
import base64
import io
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
import streamlit as st

from ml_models import FEATURES, TARGET, load_bundle, optimize_model, train_models, validate_frame
from model_guide import MODEL_GUIDES
from parity_ui import render_parity
from r2_chart import render_r2_chart
from shap_beeswarm import exact_grouped_interventional_shap
from shap_ui import is_single_ann_training, render_ann_shap


def _asset_data_uri(relative_path: str) -> str:
    """Encode a local visual asset so it renders consistently on Streamlit Cloud."""
    asset = Path(__file__).parent / relative_path
    if not asset.exists():
        return ""
    return "data:image/png;base64," + base64.b64encode(asset.read_bytes()).decode("ascii")

st.set_page_config(page_title="TGV · Dopamine Lab", page_icon="🧪", layout="wide", initial_sidebar_state="expanded")
st.markdown("""
<style>
:root { --ink:#201238; --muted:#756987; --purple:#6c2bd9; --pink:#ff3d81; --cyan:#00b8ce; --yellow:#ffd43b; --line:#e8def7; }
html, body, .stApp { font-family:"Microsoft YaHei","Noto Sans SC","Segoe UI",Arial,sans-serif; }
.stApp p, .stApp h1, .stApp h2, .stApp h3, .stApp label, .stApp button { font-family:inherit; }
.stApp { background:linear-gradient(135deg,#fff9fc 0%,#f5f0ff 52%,#effcff 100%); color:var(--ink); }
[data-testid="stHeader"] { background:transparent; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#241044 0%,#4d2097 68%,#db3476 100%); border-right:1px solid #ffffff30; }
[data-testid="stSidebar"] > div:first-child { padding-top:1.5rem; }
[data-testid="stSidebar"] label, [data-testid="stSidebar"] .stMarkdown, [data-testid="stSidebar"] .stCaption { color:#fff !important; }
[data-testid="stSidebar"] [data-testid="stFileUploader"] small { color:#f7eefe !important; }
[data-testid="stSidebar"] [data-testid="stCodeBlock"] { background:#1d1036 !important; border:1px solid #c58cff; }
[data-testid="stSidebar"] [data-testid="stCodeBlock"] code { color:#ffe66d !important; background:transparent !important; }
.particles { position:fixed; inset:0; z-index:0; pointer-events:none; overflow:hidden; }
.particle { position:absolute; width:7px; height:7px; border-radius:50%; opacity:.65; filter:blur(.2px); box-shadow:0 0 12px currentColor; animation:particleFloat var(--duration) ease-in-out infinite alternate; }
.particle:nth-child(1) { left:7%; top:25%; color:#ff3d81; background:#ff3d81; --duration:8s; }
.particle:nth-child(2) { left:19%; top:74%; color:#00c2d7; background:#00c2d7; --duration:11s; animation-delay:-3s; }
.particle:nth-child(3) { left:37%; top:15%; color:#ffd43b; background:#ffd43b; --duration:9s; animation-delay:-5s; }
.particle:nth-child(4) { left:56%; top:70%; color:#6c2bd9; background:#6c2bd9; --duration:13s; animation-delay:-2s; }
.particle:nth-child(5) { left:76%; top:24%; color:#00c2d7; background:#00c2d7; --duration:10s; animation-delay:-7s; }
.particle:nth-child(6) { left:91%; top:63%; color:#ff3d81; background:#ff3d81; --duration:12s; animation-delay:-4s; }
.particle:nth-child(7) { left:46%; top:90%; color:#ffd43b; background:#ffd43b; --duration:8s; animation-delay:-1s; }
@keyframes particleFloat { 0% { transform:translate(0,0) scale(.65); } 50% { transform:translate(28px,-32px) scale(1.25); } 100% { transform:translate(-20px,24px) scale(.8); } }
.floating-stars { position:fixed; inset:0; z-index:999; pointer-events:none; overflow:hidden; }
.floating-star { position:absolute; left:var(--x); top:var(--y); width:var(--size); height:var(--size); color:var(--color); background:var(--color); opacity:.78; clip-path:polygon(50% 0,60% 40%,100% 50%,60% 60%,50% 100%,40% 60%,0 50%,40% 40%); filter:drop-shadow(0 0 4px #fff) drop-shadow(0 0 9px currentColor); animation:starDrift var(--duration) ease-in-out infinite; animation-delay:var(--delay); will-change:transform,opacity; }
@keyframes starDrift { 0%,100% { transform:translate3d(0,0,0) rotate(0deg) scale(.7); opacity:.38; } 30% { transform:translate3d(16px,-20px,0) rotate(35deg) scale(1.15); opacity:.95; } 65% { transform:translate3d(-13px,17px,0) rotate(75deg) scale(.86); opacity:.62; } }
.main .block-container { position:relative; z-index:1; max-width:1380px; padding-top:1.65rem; padding-bottom:3rem; }
.hero { position:relative; overflow:hidden; padding:2.3rem 2.5rem; border-radius:30px; color:white; background:linear-gradient(120deg,#3d168f 0%,#782ad0 45%,#ff3d81 100%); box-shadow:0 20px 55px #5d25b62b; margin-bottom:1rem; }
.hero:after { content:""; position:absolute; width:260px; height:260px; right:-65px; top:-110px; border:42px solid #ffffff18; border-radius:50%; }
.hero h1 { position:relative; z-index:1; font-size:clamp(2rem,4vw,3rem); margin:0; letter-spacing:-.045em; line-height:1.18; }
.hero p { position:relative; z-index:1; margin:.65rem 0 0; opacity:.92; font-size:1.02rem; }
.laser-hero { min-height:210px; display:flex; align-items:center; padding-right:31%; }
.laser-hero:before { content:""; position:absolute; right:13%; bottom:-25%; width:330px; height:330px; border-radius:50%; background:radial-gradient(circle, #ff9bd04d 0%, #8e52e61f 48%, transparent 72%); filter:blur(3px); z-index:0; }
.laser-hero-copy { position:relative; z-index:3; max-width:100%; }
.laser-hero-character { position:absolute; z-index:2; right:3.5%; bottom:-7%; width:min(31%, 365px); max-height:122%; object-fit:contain; object-position:center bottom; opacity:.84; filter:drop-shadow(0 12px 18px #24063c45) saturate(1.08); pointer-events:none; }
.laser-star { position:absolute; z-index:3; width:10px; height:10px; background:#ffe3f2; clip-path:polygon(50% 0,60% 40%,100% 50%,60% 60%,50% 100%,40% 60%,0 50%,40% 40%); filter:drop-shadow(0 0 8px #fff); animation:heroStarTwinkle 3.8s ease-in-out infinite; }
.laser-star:nth-of-type(1) { right:35%; top:22%; animation-delay:-1.2s; }
.laser-star:nth-of-type(2) { right:8%; top:18%; width:7px; height:7px; animation-delay:-2.7s; }
.laser-star:nth-of-type(3) { right:29%; bottom:18%; width:6px; height:6px; animation-delay:-.4s; }
@keyframes heroStarTwinkle { 0%,100% { opacity:.25; transform:scale(.65) rotate(0deg); } 50% { opacity:.95; transform:scale(1.25) rotate(45deg); } }
.plating-hero { min-height:210px; display:flex; align-items:center; padding-right:31%; }
.plating-hero:before { content:""; position:absolute; right:10%; bottom:-35%; width:390px; height:390px; border-radius:50%; background:radial-gradient(circle, #ffb3dd45 0%, #8e52e62b 45%, transparent 73%); filter:blur(4px); z-index:0; }
.plating-hero-copy { position:relative; z-index:3; max-width:100%; }
.plating-hero-character { position:absolute; z-index:2; right:1.5%; bottom:-13%; width:min(33%, 430px); max-height:128%; object-fit:contain; object-position:center bottom; opacity:.76; filter:drop-shadow(0 14px 20px #24063c3d) saturate(1.12); pointer-events:none; }
.pill { display:inline-block; padding:.35rem .8rem; border-radius:999px; background:#ffd43b; color:#35175f; font-weight:800; font-size:.78rem; margin-bottom:.65rem; }
.workflow { display:flex; gap:.55rem; flex-wrap:wrap; margin:.2rem 0 1.15rem; }
.step { background:#ffffffb8; border:1px solid #eadff8; color:#5c4876; padding:.48rem .72rem; border-radius:12px; font-size:.82rem; font-weight:700; box-shadow:0 5px 16px #4b1fb50a; }
.step b { color:#6c2bd9; margin-right:.3rem; }
.card { background:#fffffff0; border:1px solid #eaddf7; border-radius:20px; padding:1.15rem 1.3rem; box-shadow:0 10px 32px #4b1fb512; color:var(--ink); }
.card p { color:var(--muted); line-height:1.7; }
.upload-info { min-height:108px; display:flex; flex-direction:column; justify-content:center; }
.upload-info strong { color:#4b1fb8; font-size:1.05rem; }
.upload-info span { color:#756987; margin-top:.3rem; }
.feature { background:linear-gradient(135deg,#fff,#fbf8ff); border-radius:14px 14px 5px 5px; padding:.68rem .85rem; border:1px solid #e9ddf7; border-left:5px solid var(--cyan); box-shadow:0 5px 18px #4b1fb50d; color:#382051; }
.section-title { color:#4b1fb8; font-size:1.3rem; font-weight:800; margin:1rem 0 .85rem; letter-spacing:-.02em; }
div.stButton > button { min-height:2.9rem; border:0; border-radius:13px; font-weight:800; background:linear-gradient(110deg,#6c2bd9,#ff3d81); color:white; box-shadow:0 8px 18px #6c2bd92d; transition:all .2s ease; }
div.stButton > button:hover { transform:translateY(-2px); color:white; box-shadow:0 11px 24px #6c2bd93d; }
div.stDownloadButton > button { min-height:2.9rem; border:1px solid #d7c3f1; border-radius:13px; color:#5b20ad; background:#fff; font-weight:800; }
[data-testid="stMetric"] { background:#ffffffed; border:1px solid #eadff6; border-radius:17px; padding:.9rem 1.05rem; box-shadow:0 8px 24px #4b1fb511; }
[data-testid="stMetricLabel"] { color:#786c87; }
[data-testid="stMetricValue"] { color:#4b1fb8; font-weight:800; }
[data-baseweb="tab-list"] { gap:.45rem; background:#ffffffad; padding:.42rem; border:1px solid #eadff5; border-radius:16px; box-shadow:0 8px 24px #4b1fb50b; }
[data-baseweb="tab"] { height:2.8rem; border-radius:11px; padding:0 1.1rem; color:#6e617e; font-weight:700; }
[aria-selected="true"][data-baseweb="tab"] { background:linear-gradient(105deg,#6c2bd9,#a737d0); color:#fff !important; }
[data-baseweb="tab-highlight"], [data-baseweb="tab-border"] { display:none; }
[data-testid="stFileUploader"] section { border:1.5px dashed #b99be2; border-radius:17px; background:#ffffffd9; min-height:108px; }
[data-testid="stFileUploader"] section:hover { border-color:#ff3d81; background:#fff; }
[data-testid="stDataFrame"] { border:1px solid #eadff6; border-radius:15px; overflow:hidden; }
[data-baseweb="input"] { border-radius:12px !important; border-color:#ddcef0 !important; background:#fff !important; }
[data-testid="stAlert"] { border-radius:15px; }
.algorithm-intro { padding:1.4rem 1.6rem; border-radius:20px; background:linear-gradient(120deg,#f0e5ff,#fff3f8); border:1px solid #dfcdf2; margin:.8rem 0 1.1rem; }
.algorithm-intro h3 { color:#4b1fb8; margin:0 0 .45rem; }
.algorithm-intro p { color:#5c4876; margin:0; }
[data-testid="stLatex"] { overflow-x:auto; padding:.6rem 0; }
hr { border-color:#ded2ec !important; }
@media (max-width:760px) { .main .block-container { padding-left:1rem; padding-right:1rem; } .hero { padding:1.65rem 1.3rem; border-radius:22px; } .hero h1 { font-size:2rem; } .workflow { display:grid; grid-template-columns:1fr 1fr; } [data-baseweb="tab"] { padding:0 .55rem; font-size:.78rem; } .laser-hero, .plating-hero { min-height:190px; padding-right:1.3rem; } .laser-hero-character, .plating-hero-character { width:35%; right:-2%; opacity:.38; } .laser-hero-copy, .plating-hero-copy { max-width:92%; } }
@media (max-width:520px) { .laser-hero-character, .plating-hero-character { opacity:.22; width:42%; } }
@media (prefers-reduced-motion:reduce) { .particle, .floating-star, .laser-star { animation:none; } div.stButton > button { transition:none; } }
</style>
""", unsafe_allow_html=True)
st.markdown('<div class="particles" aria-hidden="true"><span class="particle"></span><span class="particle"></span><span class="particle"></span><span class="particle"></span><span class="particle"></span><span class="particle"></span><span class="particle"></span></div>', unsafe_allow_html=True)
st.markdown('''<div class="floating-stars" aria-hidden="true">
<span class="floating-star" style="--x:5%;--y:16%;--size:14px;--color:#ff3d81;--duration:10s;--delay:-2s"></span>
<span class="floating-star" style="--x:14%;--y:46%;--size:9px;--color:#00cfe8;--duration:14s;--delay:-7s"></span>
<span class="floating-star" style="--x:9%;--y:82%;--size:18px;--color:#ffd43b;--duration:17s;--delay:-5s"></span>
<span class="floating-star" style="--x:24%;--y:25%;--size:11px;--color:#b06cff;--duration:13s;--delay:-9s"></span>
<span class="floating-star" style="--x:31%;--y:68%;--size:15px;--color:#ff70ba;--duration:16s;--delay:-3s"></span>
<span class="floating-star" style="--x:42%;--y:10%;--size:8px;--color:#45e6c1;--duration:12s;--delay:-6s"></span>
<span class="floating-star" style="--x:48%;--y:54%;--size:17px;--color:#ff9f43;--duration:18s;--delay:-12s"></span>
<span class="floating-star" style="--x:55%;--y:88%;--size:10px;--color:#7c6cff;--duration:15s;--delay:-1s"></span>
<span class="floating-star" style="--x:64%;--y:30%;--size:14px;--color:#00cfe8;--duration:11s;--delay:-8s"></span>
<span class="floating-star" style="--x:70%;--y:72%;--size:19px;--color:#ff4f9a;--duration:19s;--delay:-4s"></span>
<span class="floating-star" style="--x:78%;--y:12%;--size:9px;--color:#ffe66d;--duration:13s;--delay:-10s"></span>
<span class="floating-star" style="--x:84%;--y:48%;--size:16px;--color:#a66cff;--duration:16s;--delay:-6s"></span>
<span class="floating-star" style="--x:92%;--y:22%;--size:12px;--color:#54e1bc;--duration:14s;--delay:-11s"></span>
<span class="floating-star" style="--x:95%;--y:84%;--size:18px;--color:#ff86c8;--duration:20s;--delay:-2s"></span>
</div>''', unsafe_allow_html=True)
workspace_mode = st.radio(
    "选择研究工作区",
    ["电镀填充回归", "激光钻孔 AI"],
    horizontal=True,
    key="workspace_mode",
)
if workspace_mode == "激光钻孔 AI":
    from laser_ui import render_laser_platform

    render_laser_platform()
    st.markdown("---")
    st.caption("Dopamine Lab · TGV 激光钻孔质量评估与贝叶斯优化")
    st.stop()
plating_character_uri = _asset_data_uri("assets/plating_character.png")
plating_character = (
    f'<img class="plating-hero-character" src="{plating_character_uri}" alt="电镀填充实验室插图">'
    if plating_character_uri else ""
)
st.markdown(
    '<div class="hero plating-hero"><div class="plating-hero-copy">'
    '<span class="pill">TGV · ML STUDIO</span><h1>🧪 电镀填充智能实验室</h1>'
    '<p>六种回归模型 · 单变量输入 · 快速预测 Area_ratio</p></div>'
    + plating_character + '</div>',
    unsafe_allow_html=True,
)
st.markdown('<div class="workflow"><span class="step"><b>01</b>导入数据</span><span class="step"><b>02</b>选择模型</span><span class="step"><b>03</b>训练评估</span><span class="step"><b>04</b>输入参数预测</span></div>', unsafe_allow_html=True)

MODEL_NAMES = ["ANN", "XGBOOST", "RF", "GPR", "SVR", "PR"]
UNITS = {
    "Depth": "μm",
    "Wide": "μm",
    "i0": "A/m²",
    "c_ACC": "mol/m³",
    "c_SUP": "mol/m³",
    "c_LEVE": "mol/m³",
}
st.session_state.setdefault("models", {})
st.session_state.setdefault("metrics", pd.DataFrame())
st.session_state.setdefault("data", None)

def read_upload(file) -> pd.DataFrame:
    return pd.read_csv(file) if file.name.lower().endswith(".csv") else pd.read_excel(file)

# Keep the primary import control in the upper-left of the main page so it is
# visible even when the sidebar is collapsed.
upload_col, status_col = st.columns([1.35, 2.65])
with upload_col:
    uploaded = st.file_uploader("📁 导入实验数据（CSV / Excel）", type=["csv", "xlsx", "xls"], key="main_upload")
with status_col:
    st.markdown('<div class="card upload-info"><strong>清晰的数据，从这里开始</strong><span>支持 CSV、XLSX、XLS · 需要 6 个输入特征和 1 个目标变量 Area_ratio</span></div>', unsafe_allow_html=True)

with st.sidebar:
    st.markdown("## 🎛️ 实验控制台")
    st.caption("配置训练参数与参与比较的模型")
    st.markdown("#### 数据划分")
    test_size = st.slider("测试集比例", 0.1, 0.4, 0.2, 0.05)
    seed = st.number_input("随机种子", 0, 9999, 42)
    st.markdown("#### 模型选择")
    selection_mode = st.radio("训练方式", ["单模型训练", "多模型对比"], key="training_mode")
    if selection_mode == "单模型训练":
        active_model = st.selectbox("独立选择模型", MODEL_NAMES, format_func=lambda name: MODEL_GUIDES[name]["name"], key="single_training_model")
        selected = [active_model]
        st.caption(MODEL_GUIDES[active_model]["summary"])
    else:
        st.caption("分别勾选需要训练的模型；只勾选一个也可独立运行。")
        selected = [name for name in MODEL_NAMES if st.checkbox(MODEL_GUIDES[name]["name"], value=True, key=f"train_{name}")]
    st.caption(f"本次将训练 {len(selected)} 个模型；原有训练结果在下一次成功训练后替换。")
    st.markdown("---")
    st.caption("必需字段")
    st.code(", ".join(FEATURES + [TARGET]), language="text")
    st.caption("输入单位")
    st.markdown("孔深 Depth / 孔宽 Wide：μm  ·  i0：A/m²  ·  c_ACC / c_SUP / c_LEVE：mol/m³")

if uploaded:
    try:
        st.session_state.data = read_upload(uploaded)
        st.toast(f"已读取 {len(st.session_state.data):,} 行数据", icon="✅")
    except Exception as exc:
        st.error(f"读取文件失败：{exc}")
data = st.session_state.data

with st.expander("💾 模型存档 · 导入模型并恢复预测", expanded=True):
    st.caption("无需上传实验数据。选择之前下载的 tgv_models.joblib，导入后进入“独立输入预测”。刷新或服务器重启后可再次导入。")
    st.warning("仅导入自己保存或完全信任的模型文件：joblib 文件加载时可以执行代码。不要上传来源不明的文件。")
    model_file = st.file_uploader("选择已保存的模型文件（.joblib）", type=["joblib"], key="model_upload")
    trusted_model = st.checkbox("我确认此模型文件来源可信，并了解加载风险", key="trust_model")
    if st.button("📥 导入并恢复预测", disabled=model_file is None or not trusted_model, key="restore_models"):
        try:
            model_file.seek(0)
            bundle = load_bundle(model_file)
            # 全部校验成功后才替换会话，导入失败保留当前模型。
            st.session_state.models = bundle["models"]
            st.session_state.metrics = pd.DataFrame()
            # 导入模型并未附带独立测试数据，必须清除旧模型的图，禁止混用。
            st.session_state.pop("parity_evaluation", None)
            st.session_state.pop("parity_export", None)
            st.session_state.pop("ann_shap_explanation", None)
            st.session_state.pop("ann_shap_export", None)
            st.session_state.pop("chosen_model", None)
            st.success("模型恢复成功，可直接预测：" + "、".join(bundle["models"]))
        except Exception as exc:
            st.error(f"模型导入失败，当前模型未替换：{exc}。请确认文件完整，且训练与运行环境的依赖版本兼容。")
    if st.session_state.models:
        st.caption("当前可用模型：" + "、".join(st.session_state.models))
        archive = io.BytesIO()
        joblib.dump({"models": st.session_state.models, "features": FEATURES, "target": TARGET}, archive)
        st.download_button("💾 备份当前可用模型", archive.getvalue(), "tgv_models.joblib", key="backup_models")

tab_train, tab_predict, tab_algorithms, tab_optuna, tab_help = st.tabs(["📊 训练与评估", "🎯 独立输入预测", "📚 算法原理", "✨ Optuna 优化", "💡 使用说明"])

with tab_train:
    st.caption("本次训练选择：" + (" · ".join(selected) if selected else "尚未选择模型"))
    if data is None:
        st.markdown('<div class="card"><h3>👋 从数据开始</h3><p>请使用页面左上方的数据入口，上传包含 6 个输入特征和 Area_ratio 的数据文件。</p></div>', unsafe_allow_html=True)
        sample_path = Path(__file__).with_name("sample_data.csv")
        if sample_path.exists():
            st.download_button("⬇️ 下载示例数据", sample_path.read_bytes(), "sample_data.csv", "text/csv", width="content")
    else:
        st.markdown('<div class="section-title">数据概览</div>', unsafe_allow_html=True)
        a, b, c = st.columns(3)
        a.metric("数据行数", f"{len(data):,}")
        b.metric("输入特征", "6")
        c.metric("目标变量", TARGET)
        with st.expander("展开查看数据预览", expanded=True):
            # Streamlit 的 dataframe 组件支持虚拟滚动，因此可以安全地展示
            # 全部上传行，而不会把 1000+ 行一次性渲染成静态 HTML。
            preview_height = min(720, max(420, 120 + len(data) * 28))
            st.caption(f"已载入全部 {len(data):,} 行，未截取数据；在表格内上下滚动即可查看，也可点击列标题排序。")
            st.dataframe(
                data,
                width="stretch",
                height=preview_height,
                hide_index=False,
                column_config={
                    "Depth": st.column_config.NumberColumn("孔深 Depth (μm)"),
                    "Wide": st.column_config.NumberColumn("孔宽 Wide (μm)"),
                    "i0": st.column_config.NumberColumn("i0 (A/m²)"),
                    "c_ACC": st.column_config.NumberColumn("c_ACC (mol/m³)"),
                    "c_SUP": st.column_config.NumberColumn("c_SUP (mol/m³)"),
                    "c_LEVE": st.column_config.NumberColumn("c_LEVE (mol/m³)"),
                    TARGET: st.column_config.NumberColumn("Area_ratio"),
                },
            )
            st.download_button(
                "⬇️ 下载已上传的完整数据",
                data.to_csv(index=False).encode("utf-8-sig"),
                "tgv_uploaded_data.csv",
                "text/csv",
                width="content",
                key="download_uploaded_data",
            )
        left, right = st.columns([1, 1])
        with left:
            if st.button(f"🚀 训练 {selected[0]}" if len(selected) == 1 else f"🚀 对比训练 {len(selected)} 个模型", type="primary", width="stretch", disabled=not selected):
                with st.spinner("模型训练中，请稍候…"):
                    try:
                        models, metrics, x_test, y_test = train_models(data, test_size, int(seed), selected)
                        # 所有图均使用本轮留出的同一测试集，不重新划分或拟合。
                        evaluation = {
                            "y_true": y_test.to_numpy(copy=True),
                            "predictions": {name: model.predict(x_test) for name, model in models.items()},
                            "source_index": y_test.index.tolist(),
                            "seed": int(seed), "test_size": test_size,
                        }
                        ann_explanation = None
                        if is_single_ann_training(selection_mode, selected):
                            # 解释测试集；背景仅来自训练部分，避免把测试样本作为参考基线。
                            clean = validate_frame(data)
                            background = clean.loc[
                                ~clean.index.isin(x_test.index), FEATURES
                            ]
                            if background.empty:
                                raise ValueError("无法构建 ANN SHAP 的训练集背景样本。")
                            if (x_test["Wide"] == 0).any():
                                raise ValueError("Wide 含 0，无法计算 Aspect ratio = Depth / Wide。")
                            shap_values, source_values, expected_value = exact_grouped_interventional_shap(
                                models["ANN"], x_test[FEATURES].to_numpy(),
                                background.to_numpy(), feature_names=FEATURES,
                                feature_groups=((0, 1), (2,), (3,), (4,), (5,)),
                                max_samples=500, max_background=16,
                                random_state=int(seed),
                            )
                            explained_values = np.column_stack([
                                source_values[:, 0] / source_values[:, 1],
                                source_values[:, 2:],
                            ])
                            ann_explanation = {
                                "shap_values": shap_values,
                                "feature_values": explained_values,
                                "source_feature_values": source_values,
                                "feature_names": [
                                    "Aspect ratio", "i0", "c_ACC", "c_SUP", "c_LEVE"
                                ],
                                "output_name": TARGET,
                                "expected_value": expected_value,
                                "background_size": min(len(background), 16),
                                "seed": int(seed),
                            }
                        st.session_state.models, st.session_state.metrics = models, metrics
                        st.session_state.parity_evaluation = evaluation
                        st.session_state.pop("parity_export", None)
                        if ann_explanation is not None:
                            st.session_state.ann_shap_explanation = ann_explanation
                        else:
                            st.session_state.pop("ann_shap_explanation", None)
                        st.session_state.pop("ann_shap_export", None)
                        st.success("训练完成！")
                    except Exception as exc:
                        st.error(f"训练失败：{exc}")
        with right:
            if st.session_state.models:
                buffer = io.BytesIO()
                joblib.dump({"models": st.session_state.models, "features": FEATURES, "target": TARGET}, buffer)
                st.download_button("💾 下载训练模型", buffer.getvalue(), "tgv_models.joblib", width="stretch")
        if not st.session_state.metrics.empty:
            st.markdown('<div class="section-title">模型表现</div>', unsafe_allow_html=True)
            best = st.session_state.metrics.iloc[0]
            m1, m2, m3 = st.columns(3)
            m1.metric("当前最佳模型", best["Model"])
            m2.metric("最佳 R²", f"{best['R2']:.4f}")
            m3.metric("对应 RMSE", f"{best['RMSE']:.6g}")
            st.dataframe(st.session_state.metrics.style.format({"R2":"{:.4f}", "RMSE":"{:.6g}", "MAE":"{:.6g}"}), width="stretch")
            render_r2_chart(st.session_state.metrics)
    render_parity()
    if is_single_ann_training(selection_mode, selected):
        render_ann_shap()

with tab_predict:
    st.markdown('<div class="section-title">逐项填写实验条件</div>', unsafe_allow_html=True)
    st.caption("每个输入变量都可以单独填写；单位已标注在变量名中，请与训练数据保持一致。")
    if not st.session_state.models:
        st.warning("请先训练模型，或在上方“模型存档”中导入已保存的模型。")
    else:
        chosen = st.selectbox("选择预测模型", list(st.session_state.models), key="chosen_model")
        defaults = {"Depth":20.0, "Wide":24.0, "i0":2.0, "c_ACC":10.0, "c_SUP":50.0, "c_LEVE":5.0}
        values = {}
        cols = st.columns(3)
        for i, feature in enumerate(FEATURES):
            with cols[i % 3]:
                label = f"{ {'Depth': '孔深 Depth', 'Wide': '孔宽 Wide'}.get(feature, feature)}（{UNITS[feature]}）"
                st.markdown(f'<div class="feature"><b>{label}</b></div>', unsafe_allow_html=True)
                values[feature] = st.number_input(f"输入 {label}", value=defaults[feature], format="%.8g", key=f"input_{feature}", label_visibility="collapsed", help=f"单位：{UNITS[feature]}")
        if st.button("🎯 计算 Area_ratio", type="primary", width="stretch"):
            prediction = st.session_state.models[chosen].predict(pd.DataFrame([values]))[0]
            st.metric("预测填充面积比 Area_ratio", f"{prediction:.8g}")

with tab_algorithms:
    st.markdown('<div class="section-title">了解模型，再选择模型</div>', unsafe_allow_html=True)
    st.caption("无需上传数据即可查阅。以下公式描述通用回归算法；工艺用途为建议，实际精度以独立验证为准。")
    guide_model = st.radio("独立查看算法", MODEL_NAMES, horizontal=True,
                           index=MODEL_NAMES.index(selected[0]) if selected else 0, key="guide_model")
    guide = MODEL_GUIDES[guide_model]
    st.markdown(f'<div class="algorithm-intro"><h3>{guide["name"]}</h3><p>{guide["family"]} · {guide["summary"]}</p></div>', unsafe_allow_html=True)
    theory_col, practice_col = st.columns([1.45, 1])
    with theory_col:
        st.markdown("#### 算法原理")
        st.write(guide["principle"])
        st.markdown("#### 核心数学公式")
        for equation in guide["equations"]:
            st.latex(equation)
        st.caption(guide["symbols"])
        st.caption("通用符号：x=(Depth, Wide, i0, c_ACC, c_SUP, c_LEVE)，y=Area_ratio，ŷ 为预测值。")
    with practice_col:
        st.markdown("#### 擅长解决的问题")
        st.info(guide["strengths"])
        st.markdown("#### 在 TGV 电镀中的应用")
        st.write(guide["application"])
        st.markdown("#### 适用边界与注意事项")
        st.warning(guide["limitations"])
    with st.expander("查看当前代码的初始参数（不是调参最优值）"):
        from ml_models import build_models
        estimator = build_models(int(seed))[guide_model]
        if estimator is None:
            st.info("XGBoost 依赖未安装；安装后可读取当前参数。")
        else:
            params = estimator.get_params(deep=True)
            keys = {
                "ANN": ["model__hidden_layer_sizes", "model__activation", "model__learning_rate_init", "model__batch_size", "model__max_iter"],
                "XGBOOST": ["max_depth", "n_estimators", "learning_rate", "objective"],
                "RF": ["n_estimators", "max_depth", "max_features", "bootstrap"],
                "GPR": ["model__kernel", "model__normalize_y", "model__n_restarts_optimizer"],
                "SVR": ["model__kernel", "model__gamma", "model__C", "model__epsilon"],
                "PR": ["poly__degree", "poly__include_bias", "model__fit_intercept"],
            }[guide_model]
            st.dataframe(pd.DataFrame([{"API 参数": key, "当前值": str(params[key])} for key in keys]), hide_index=True, width="stretch")
    st.markdown("#### 官方参考资料")
    for label, url in guide["sources"]:
        st.markdown(f"- [{label}]({url})")
    st.caption("资料查阅：2026-11-01。说明与本项目现有六种回归模型对应；未引入额外分类算法。")

with tab_optuna:
    st.markdown('<div class="section-title">自动寻找更优超参数</div>', unsafe_allow_html=True)
    if data is None:
        st.warning("请先上传数据文件。")
    else:
        oc1, oc2, oc3 = st.columns([1, 1, 1])
        opt_model = oc1.selectbox("优化模型", ["ANN", "XGBOOST"])
        trials = oc2.number_input("试验次数", 5, 100, 20)
        if oc3.button("✨ 开始优化", type="primary", width="stretch"):
            with st.spinner("Optuna 搜索中…"):
                try:
                    params, score = optimize_model(data, opt_model, int(trials), test_size, int(seed))
                    st.success(f"最佳验证 RMSE：{score:.6g}")
                    st.json(params)
                except Exception as exc:
                    st.error(str(exc))

with tab_help:
    st.markdown('<div class="card"><h3>📌 数据格式</h3><p>CSV 或 Excel 表头必须包含：</p><code>Depth, Wide, i0, c_ACC, c_SUP, c_LEVE, Area_ratio</code><p>单位：Depth/Wide 为 μm；i0 为 A/m²；c_ACC、c_SUP、c_LEVE 为 mol/m³。</p><h3>🧭 推荐流程</h3><p>上传数据 → 选择模型 → 开始训练 → 查看 R²/RMSE/MAE → 在“独立输入预测”中逐项填入实验条件。</p><h3>⚠️ 研究提示</h3><p>正式论文建议增加交叉验证、独立外部测试集和异常值审查。GPR 与五阶 PR 在大数据集上可能较慢。</p></div>', unsafe_allow_html=True)
st.markdown("---")
st.caption("Dopamine Lab · TGV 电镀填充机器学习框架")
