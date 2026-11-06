"""Independent dopamine-themed Streamlit workspace for the P0 GNN prototype."""
from __future__ import annotations

import streamlit as st

try:  # Cloud entrypoint imports this as a top-level module.
    from .gnn_web import (
        export_graphs_as_poscar_zip,
        load_model_bundle,
        parse_graph_csv,
        predict_graphs,
        validate_adsorption_csv,
        validate_candidate_csv,
    )
except ImportError:  # Streamlit executes cloud_deploy/app.py from this directory.
    from gnn_web import (
        export_graphs_as_poscar_zip,
        load_model_bundle,
        parse_graph_csv,
        predict_graphs,
        validate_adsorption_csv,
        validate_candidate_csv,
    )


CANDIDATE_TEMPLATE = (
    "molecule_id,canonical_smiles,additive_role,formal_charge,protonation_state,"
    "counterion_policy,chemical_family,source\n"
    "candidate_001,CS(=O)(=O)O,accelerator,0,neutral,explicit,sulfur-containing,user\n"
).encode("utf-8-sig")
ADSORPTION_TEMPLATE = (
    "molecule_id,config_id,facet,E_slab_adsorbate,E_clean_slab,E_isolated_molecule,"
    "E_ads,energy_unit,calculation_version\n"
    "candidate_001,config_0000,Cu111,-100.0,-90.0,-9.2,-0.8,eV,protocol_v1\n"
).encode("utf-8-sig")
GRAPH_TEMPLATE = (
    'molecule_id,symbols,positions_angstrom,cell_angstrom,pbc,atom_roles,config_id,facet,family\n'
    'example_001,"[""Cu"",""S""]","[[0,0,0],[0,0,1.8]]",'
    '"[[8,0,0],[0,8,0],[0,0,12]]","[true,true,false]","[1,0]",config_0000,Cu111,example\n'
).encode("utf-8-sig")


def _inject_gnn_styles() -> None:
    st.markdown("""
<style>
.gnn-hero{min-height:235px;display:flex;align-items:center;padding-right:34%;background:linear-gradient(118deg,#291065 0%,#6634bd 47%,#00a9bd 100%)}
.gnn-hero:before{content:"";position:absolute;right:3%;top:-25%;width:390px;height:390px;border-radius:50%;background:radial-gradient(circle,#8cf6e755 0%,#dd9aff26 44%,transparent 72%)}
.gnn-copy{position:relative;z-index:3}.gnn-network{position:absolute;right:4.5%;top:13%;width:27%;height:74%;z-index:2;opacity:.9}
.gnn-node{position:absolute;width:27px;height:27px;border-radius:50%;background:#fff;border:5px solid var(--node);box-shadow:0 0 20px var(--node);animation:gnnPulse 3.6s ease-in-out infinite}
.gnn-node.n1{left:10%;top:43%;--node:#ffd43b}.gnn-node.n2{left:45%;top:8%;--node:#ff70ba;animation-delay:-1s}.gnn-node.n3{left:72%;top:45%;--node:#61f1db;animation-delay:-2s}.gnn-node.n4{left:43%;top:77%;--node:#b889ff;animation-delay:-.4s}.gnn-node.n5{left:42%;top:43%;width:37px;height:37px;--node:#fff;animation-delay:-1.7s}
.gnn-edge{position:absolute;height:3px;transform-origin:left center;background:linear-gradient(90deg,#ffffffcc,#80f6e8cc);border-radius:4px;box-shadow:0 0 8px #fff8}.gnn-edge.e1{left:17%;top:51%;width:39%;transform:rotate(-39deg)}.gnn-edge.e2{left:52%;top:23%;width:34%;transform:rotate(43deg)}.gnn-edge.e3{left:55%;top:58%;width:30%;transform:rotate(-3deg)}.gnn-edge.e4{left:53%;top:59%;width:27%;transform:rotate(84deg)}.gnn-edge.e5{left:19%;top:55%;width:34%;transform:rotate(39deg)}
@keyframes gnnPulse{0%,100%{transform:scale(.84)}50%{transform:scale(1.12)}}
.gnn-status-grid{display:grid;grid-template-columns:repeat(4,1fr);gap:.7rem;margin:.75rem 0 1rem}.gnn-status{padding:.85rem;border-radius:16px;background:#ffffffd9;border:1px solid #dcd0f2;box-shadow:0 7px 21px #421c880d;color:#655276;text-align:center}.gnn-status b{display:block;color:#4d1ca0;font-size:1.02rem;margin-bottom:.15rem}
.gnn-flow{display:grid;grid-template-columns:repeat(4,1fr);gap:.65rem;margin:.8rem 0 1.15rem}.gnn-flow-card{min-height:116px;padding:1rem;border-radius:17px;color:#4c3861;background:linear-gradient(145deg,#fff,#faf6ff);border:1px solid #e3d6f2;box-shadow:0 8px 24px #4b1fb50d}.gnn-flow-card b{display:block;color:#5d26b3;margin-bottom:.35rem}.gnn-flow-card em{font-style:normal;color:#ff3d81;font-weight:900}
.gnn-principle{min-height:165px;padding:1.1rem 1.2rem;border-radius:18px;background:#ffffffdf;border:1px solid #e2d5f1;box-shadow:0 8px 24px #4b1fb50d}.gnn-principle h4{color:#5420aa;margin:.1rem 0 .45rem}.gnn-principle p{color:#675775;line-height:1.65}
@media(max-width:850px){.gnn-status-grid,.gnn-flow{grid-template-columns:1fr 1fr}.gnn-hero{padding-right:1.3rem}.gnn-network{opacity:.25;width:40%;right:-4%}}
</style>""", unsafe_allow_html=True)


def _hero() -> None:
    st.markdown("""
<div class="hero gnn-hero"><div class="gnn-copy"><span class="pill">TGV · GNN P0 RESEARCH LAB</span>
<h1>🧬 添加剂 GNN 智能预筛实验室</h1><p>分子—铜表面图学习 · GraphSAGE-GMT · 不确定度与 DFT 闭环</p></div>
<div class="gnn-network" aria-hidden="true"><i class="gnn-edge e1"></i><i class="gnn-edge e2"></i><i class="gnn-edge e3"></i><i class="gnn-edge e4"></i><i class="gnn-edge e5"></i><i class="gnn-node n1"></i><i class="gnn-node n2"></i><i class="gnn-node n3"></i><i class="gnn-node n4"></i><i class="gnn-node n5"></i></div></div>
<div class="gnn-status-grid"><div class="gnn-status"><b>19 种元素</b>GAME-Net 兼容词表</div><div class="gnn-status"><b>PBC 图</b>周期性半径建边</div><div class="gnn-status"><b>Cu(111)</b>P0 主目标晶面</div><div class="gnn-status"><b>CPU 推理</b>云端轻量运行</div></div>
""", unsafe_allow_html=True)


def _render_data_contract() -> None:
    st.markdown("### ① 候选与数据契约检查")
    st.caption("先检查分子身份、角色、电荷、质子化态、对离子策略和 DFT 标签；检查不会训练模型。")
    left, right = st.columns(2)
    with left:
        candidate_file = st.file_uploader("候选分子 CSV", type=["csv"], key="gnn_candidates")
        st.download_button("下载候选模板", CANDIDATE_TEMPLATE, "gnn_candidates_template.csv", "text/csv", key="gnn_candidate_template")
        if st.button("校验候选数据", disabled=candidate_file is None, key="validate_gnn_candidates"):
            try:
                clean = validate_candidate_csv(candidate_file.getvalue())
                st.session_state["gnn_candidates_validated"] = clean
                st.success(f"候选数据契约通过：{len(clean)} 个唯一分子。")
            except Exception as exc:
                st.error(f"候选数据校验失败：{exc}")
        if st.session_state.get("gnn_candidates_validated") is not None:
            st.dataframe(st.session_state["gnn_candidates_validated"], use_container_width=True, hide_index=True)
    with right:
        label_file = st.file_uploader("Cu 吸附 DFT 标签 CSV", type=["csv"], key="gnn_ads_labels")
        st.download_button("下载 DFT 标签模板", ADSORPTION_TEMPLATE, "gnn_adsorption_template.csv", "text/csv", key="gnn_ads_template")
        if st.button("校验吸附标签", disabled=label_file is None, key="validate_gnn_labels"):
            try:
                clean = validate_adsorption_csv(label_file.getvalue())
                st.session_state["gnn_labels_validated"] = clean
                st.success(f"吸附标签契约通过：{len(clean)} 条；单位统一为 eV。")
            except Exception as exc:
                st.error(f"吸附标签校验失败：{exc}")
        if st.session_state.get("gnn_labels_validated") is not None:
            st.dataframe(st.session_state["gnn_labels_validated"], use_container_width=True, hide_index=True)
    st.info("同一母体的构象、质子化态和吸附构型必须共享 parent_molecule_id，并只能进入同一 CV 分组，防止数据泄漏。")


def _render_structure_and_dft() -> None:
    st.markdown("### ② 结构图检查与 DFT 输入导出")
    st.caption("每行表示一个分子—Cu 表面吸附构型；坐标和晶胞单位均为 Å。这里只建图与导出，不执行 VASP。")
    st.download_button("⬇️ 下载结构 CSV 示例", GRAPH_TEMPLATE, "tgv_gnn_graph_template.csv", "text/csv", key="gnn_graph_template")
    upload = st.file_uploader("上传吸附构型结构 CSV（≤50 条）", type=["csv"], key="gnn_structure_check")
    if st.button("检查 PBC 图结构", disabled=upload is None, key="check_gnn_graphs"):
        try:
            graphs, metadata = parse_graph_csv(upload.getvalue())
            st.session_state["gnn_checked_graphs"] = graphs
            st.session_state["gnn_graph_metadata"] = metadata
            st.success(f"结构图检查完成：{len(graphs)} 个构型。")
        except Exception as exc:
            st.error(f"结构图检查失败：{exc}")
    metadata = st.session_state.get("gnn_graph_metadata")
    graphs = st.session_state.get("gnn_checked_graphs")
    if metadata is not None:
        m1, m2, m3 = st.columns(3)
        m1.metric("构型数", len(metadata)); m2.metric("原子总数", int(metadata["n_atoms"].sum())); m3.metric("双向边总数", int(metadata["n_edges"].sum()))
        st.dataframe(metadata, use_container_width=True, hide_index=True)
    if graphs:
        try:
            poscar_zip = export_graphs_as_poscar_zip(graphs)
            st.download_button("📦 下载 POSCAR + sidecar JSON", poscar_zip, "tgv_gnn_poscar_batch.zip", "application/zip", key="gnn_poscar_zip")
        except Exception as exc:
            st.error(f"POSCAR 导出失败：{exc}")
    st.warning("POSCAR 是外部 DFT 的起点，不代表构型已弛豫或能量已收敛；泛函、色散、赝势、截断能、k 点与溶剂模型必须统一记录。")


def _render_inference() -> None:
    st.markdown("### ③ 导入模型并进行轻量推理")
    st.warning("仓库不内置未经验证的权重。必须导入您在本地/计算服务器训练的模型包，网页才会预测。")
    model_upload = st.file_uploader("导入已训练模型包（ZIP，≤100 MB）", type=["zip"], key="gnn_model_zip")
    if st.button("🔐 安全加载模型", disabled=model_upload is None, key="load_gnn_model"):
        try:
            model, card = load_model_bundle(model_upload.getvalue())
            st.session_state["gnn_model"] = model; st.session_state["gnn_model_card"] = card; st.session_state.pop("gnn_results", None)
            st.success("模型包加载成功。")
        except Exception as exc:
            st.session_state.pop("gnn_model", None); st.session_state.pop("gnn_model_card", None); st.error(f"模型包加载失败：{exc}")
    model = st.session_state.get("gnn_model"); card = st.session_state.get("gnn_model_card")
    if card:
        c1, c2, c3 = st.columns(3)
        c1.metric("模型", card.get("model_type", "unknown")); c2.metric("Pooling", card.get("pooling", "unknown")); c3.metric("不确定度", "未提供" if "unavailable" in card.get("uncertainty", "") else "可用")
        with st.expander("查看完整模型卡", expanded=False): st.json(card)
    graph_upload = st.file_uploader("上传待预测结构 CSV", type=["csv"], key="gnn_graph_csv")
    graph_schema = card.get("graph_schema", {}) if card else {}; cutoff = float(graph_schema.get("cutoff_angstrom", 3.5)); expected_pbc = graph_schema.get("pbc", [True, True, False])
    st.caption(f"当前建图：cutoff={cutoff:g} Å，PBC={expected_pbc}；必须与训练模型一致。")
    if st.button("🧪 校验并推理", disabled=model is None or graph_upload is None, key="predict_gnn"):
        try:
            graphs, metadata = parse_graph_csv(graph_upload.getvalue(), cutoff_angstrom=cutoff, expected_pbc=expected_pbc)
            with st.spinner(f"CPU 图推理中（{len(graphs)} 个结构）…"): predictions = predict_graphs(model, graphs)
            result = metadata.copy(); result["predicted_E_ads_eV"] = predictions; result["uncertainty_status"] = "not calibrated"; st.session_state["gnn_results"] = result
            st.success(f"完成 {len(result)} 个结构的推理。")
        except Exception as exc: st.error(f"结构校验或推理失败：{exc}")
    result = st.session_state.get("gnn_results")
    if result is not None:
        st.dataframe(result, use_container_width=True, hide_index=True)
        st.download_button("⬇️ 下载预测结果", result.to_csv(index=False).encode("utf-8-sig"), "tgv_gnn_p0_predictions.csv", "text/csv", key="download_gnn_results")
        st.info("单模型不提供可靠认知不确定度；低能候选必须经过统一协议 DFT 与电化学/TGV 实验复核。")


def _render_principles() -> None:
    st.markdown("### 模型原理：分子—铜表面为什么可以表示成图？")
    a, b = st.columns(2)
    with a:
        st.markdown('<div class="gnn-principle"><h4>1 · 周期性原子图</h4><p>原子是节点，邻近原子关系是双向边；节点包含元素信息，边保存原子距离。Cu 表面在 x/y 方向采用周期边界，避免把晶胞边界误当成真实断裂。</p></div>', unsafe_allow_html=True)
        st.latex(r"G=(V,E),\quad e_{ij}=1\;\mathrm{if}\;d_{ij}^{\mathrm{PBC}}\le r_c")
        st.markdown('<div class="gnn-principle"><h4>3 · GMT 图级汇聚</h4><p>Graph Multiset Transformer 通过注意力从不同原子贡献中形成固定长度的构型表示；它不是简单平均池化。</p></div>', unsafe_allow_html=True)
        st.latex(r"\alpha_i=\operatorname{softmax}(q^T k_i/\sqrt d),\quad h_G=\sum_i\alpha_i v_i")
    with b:
        st.markdown('<div class="gnn-principle"><h4>2 · GraphSAGE 消息传递</h4><p>每一层将中心原子表示与邻居聚合表示组合，使模型学习局部配位、吸附基团与铜表面的结构环境。</p></div>', unsafe_allow_html=True)
        st.latex(r"h_v^{(l+1)}=\sigma\!\left(W_s h_v^{(l)}+W_n\,\mathrm{MEAN}_{u\in\mathcal N(v)} h_u^{(l)}\right)")
        st.markdown('<div class="gnn-principle"><h4>4 · 吸附能回归与验证</h4><p>MLP 预测单构型吸附能。负值通常表示放热吸附，但“越负越好”并不成立：加速剂、抑制剂和整平剂需要不同的标定窗口。</p></div>', unsafe_allow_html=True)
        st.latex(r"E_{ads}=E_{slab+mol}-E_{clean\,slab}-E_{isolated\,mol}")
    st.markdown("#### 训练与可信度")
    st.markdown("训练时按 `parent_molecule_id` 分组执行 nested CV，目标标准化器只在训练折拟合。多个独立随机种子模型可用均值作为预测、标准差作为认知不确定度近似；适用域用训练嵌入上的正则化 Mahalanobis 距离判断。P0 单模型网页未校准这两项时会明确标记。")
    st.latex(r"\hat y=\frac1M\sum_{m=1}^{M}\hat y_m,\qquad \sigma_{epi}=\sqrt{\frac1M\sum_m(\hat y_m-\hat y)^2}")


def _render_workflow() -> None:
    st.markdown("### 从候选分子到 TGV 实验的完整闭环")
    st.markdown("""
<div class="gnn-flow"><div class="gnn-flow-card"><em>01</em><b>候选定义</b>SMILES、角色、电荷、质子化态、对离子和化学家族。</div><div class="gnn-flow-card"><em>02</em><b>吸附构型</b>在 Cu(111) 上生成多构型，保留晶胞、坐标和原子角色。</div><div class="gnn-flow-card"><em>03</em><b>统一 DFT</b>相同泛函与数值协议计算吸附能，记录全部元数据。</div><div class="gnn-flow-card"><em>04</em><b>分组训练</b>本地执行 nested CV、早停、模型集成和 OOD 标定。</div><div class="gnn-flow-card"><em>05</em><b>云端预筛</b>上传模型包与结构图，获得候选排序和风险提示。</div><div class="gnn-flow-card"><em>06</em><b>Top-K 复核</b>下载 POSCAR，在计算服务器完成弛豫和高精度 DFT。</div><div class="gnn-flow-card"><em>07</em><b>电化学实验</b>CV/EIS/RDE、吸附与界面行为验证，排除副反应。</div><div class="gnn-flow-card"><em>08</em><b>TGV 填充</b>评价 void、seam、pinch-off 与填充形貌并回填标签。</div></div>
""", unsafe_allow_html=True)
    st.markdown("#### 三类添加剂不能使用同一条简单规则")
    r1, r2, r3 = st.columns(3)
    r1.markdown("**加速剂 Accelerator**\n\n关注可参与界面反应的适中吸附与氧化还原活性，目标是促进孔底沉积，而不是单纯追求最强吸附。")
    r2.markdown("**抑制剂 Suppressor**\n\n需要稳定覆盖与传质/竞争吸附行为；分子量、构象和与 Cl⁻/加速剂协同都可能重要。")
    r3.markdown("**整平剂 Leveler**\n\n强调孔口/高电流区域选择性抑制及稳定性，需要结合电位、浓度、流动条件和真实填充结果。")
    st.warning("GNN 静态图不能直接描述 Langmuir 竞争吸附动力学、浓度效应、流场或多添加剂协同；这些属于后续 CDA/工艺代理模型与实验闭环。")


def render_gnn_workspace() -> None:
    _inject_gnn_styles(); _hero()
    st.warning("当前是 P0 候选预筛研究原型，不包含经验证的 TGV 添加剂模型权重。云端只做数据检查、POSCAR 准备和轻量推理；大型训练、构型搜索与 VASP/DFT 在本地或计算服务器执行。")
    tabs = st.tabs(["🧾 数据契约", "⚛️ 结构与 DFT", "🧠 模型推理", "📐 模型原理", "🔄 实验流程"])
    with tabs[0]: _render_data_contract()
    with tabs[1]: _render_structure_and_dft()
    with tabs[2]: _render_inference()
    with tabs[3]: _render_principles()
    with tabs[4]: _render_workflow()
    st.markdown("---")
    st.caption("P0 科学边界：输出只用于候选优先级排序；模型适用域外样本必须警告，最低能构型必须经 DFT 复核，最终结论必须来自电化学与 TGV 填充实验。")
