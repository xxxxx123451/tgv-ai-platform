"""Streamlit UI for TGV laser-drilling quality assessment and optimization."""
from __future__ import annotations

import importlib.util
from io import BytesIO
from datetime import datetime, timezone
from zoneinfo import ZoneInfo
import zipfile

import pandas as pd
import streamlit as st

from laser_optimization import (
    DEFAULT_CQS_WEIGHTS,
    AUGMENTATION_METHODS,
    LASER_PARAMETER_NAMES,
    LASER_SPACE_SPECS,
    QUALITY_METRIC_DESCRIPTIONS,
    QUALITY_METRICS,
    IterativeBayesianOptimizer,
    augment_labeled_sem_records,
    augment_sem_image,
    calculate_cqs,
    predict_sem_image,
    prepare_labeled_sem_records,
    train_quality_model,
)


def _dependency_status() -> tuple[bool, bool]:
    return importlib.util.find_spec("torch") is not None, importlib.util.find_spec("skopt") is not None


def _format_history_datetime(value: str) -> str:
    """Show stored timestamps as China Standard Time with second precision."""
    try:
        timestamp = datetime.fromisoformat(value)
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y年%m月%d日 %H时%M分%S秒")
    except (TypeError, ValueError):
        return str(value)


def render_laser_platform() -> None:
    st.markdown(
        '<div class="hero"><span class="pill">TGV · LASER AI</span>'
        '<h1>🔬 激光钻孔质量与参数优化</h1>'
        '<p>Residual U-Net 评估 SEM 图像 · 贝叶斯优化迭代建议真实实验参数</p></div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="workflow"><span class="step"><b>01</b>准备真实 SEM 与标签</span>'
        '<span class="step"><b>02</b>训练质量评估网络</span>'
        '<span class="step"><b>03</b>计算 CQS</span>'
        '<span class="step"><b>04</b>实验回填与参数迭代</span></div>',
        unsafe_allow_html=True,
    )
    st.link_button(
        "📄 打开 IEEE Xplore 原始文献",
        "https://ieeexplore.ieee.org/document/10640121",
    )
    torch_ok, skopt_ok = _dependency_status()
    model_tab, cqs_tab, bo_tab, theory_tab = st.tabs(
        ["🧠 Residual U-Net", "🧮 CQS 计算器", "🎯 贝叶斯优化", "📖 文献原理"]
    )

    with model_tab:
        st.markdown('<div class="section-title">SEM 图像 → 六项质量指标</div>', unsafe_allow_html=True)
        st.info("该模块是图像深度学习模型，与 D1/D2 等表格特征的六种回归模型相互独立。")
        st.markdown("#### ① 上传真实 SEM 图像和专家标签")
        image_files = st.file_uploader(
            "上传 SEM 图像（PNG / JPG / TIFF，可多选）",
            type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"],
            accept_multiple_files=True,
            key="sem_training_images",
            help="每张图像需在标签 CSV 中有对应的文件名和六项 0~1 标签。",
        )
        labels_file = st.file_uploader(
            "上传质量标签 CSV",
            type=["csv"],
            key="sem_quality_labels",
            help="表头必须包含 filename,NUI,SSI,TI,MCI,RI,ARI；filename 与图像文件名匹配。",
        )
        st.code("filename,NUI,SSI,TI,MCI,RI,ARI\nsem_001.tif,0.92,0.88,0.81,0.95,0.08,0.86", language="text")
        records = None
        if image_files and labels_file:
            try:
                labels_frame = pd.read_csv(labels_file)
                records = prepare_labeled_sem_records(image_files, labels_frame)
                st.success(f"已配对 {len(records)} 张图像与真实标注。")
            except Exception as exc:
                st.error(f"数据检查失败：{exc}")

        st.markdown("#### ② 数据增强：预览并生成转换图像")
        st.caption("文献中的几何变化用于模拟成像位置差异；平移按文献重点描述实现为垂直方向。亮度和高斯噪声作用于归一化灰度强度。增强图像与原图分开保存。")
        method_keys = list(AUGMENTATION_METHODS)
        selected_methods = st.multiselect(
            "选择增强方法",
            options=method_keys,
            default=method_keys,
            format_func=lambda key: AUGMENTATION_METHODS[key],
            key="sem_augmentation_methods",
        )
        aug_cols = st.columns(4)
        max_rotation = aug_cols[0].slider("旋转角度上限（°）", 0.0, 5.0, 5.0, 0.5, key="aug_rotation_max")
        max_shift = aug_cols[1].slider("垂直平移范围（像素）", 0, 50, 10, key="aug_translation_max")
        brightness_bound = aug_cols[2].slider("亮度 β 范围（±）", 0.0, 0.2, 0.2, 0.01, key="aug_brightness_bound")
        sigma_min, sigma_max = aug_cols[3].slider(
            "高斯噪声 σ 范围", 0.01, 0.05, (0.01, 0.05), 0.005, key="aug_noise_sigma"
        )
        aug_seed = st.number_input("增强随机种子", 0, 999999, 42, key="aug_seed")
        st.latex(r"G_{\mu,\sigma}(I)=I+\mathcal{N}(\mu,\sigma),\quad \mu=0,\quad \sigma\in[0.01,0.05]")
        st.caption("高斯噪声在每个像素上独立采样；输入灰度先归一化到 [0,1]，执行 I′=clip(I+noise,0,1)。亮度采用 I′=clip(I+β,0,1)，β∈[-0.2,0.2]。")
        st.caption("旋转控件按文献中描述的 0–5° 小角度方案实现；文献另有 90°/180°/270° 旋转扩增描述，本页依照本次指定采用小角度方案。")
        augmentation_options = {
            "max_rotation_degrees": float(max_rotation),
            "max_translation_pixels": int(max_shift),
            "brightness_beta": float(brightness_bound),
            "gaussian_sigma": (float(sigma_min), float(sigma_max)),
            "seed": int(aug_seed),
        }
        current_aug_signature = (
            tuple(file.name for file in image_files) if image_files else (),
            tuple(selected_methods),
            tuple(augmentation_options.items()),
        )
        if st.button(
            "✨ 生成增强图像与预览",
            key="generate_sem_augmentations",
            disabled=not image_files or not selected_methods,
        ):
            try:
                preview_records = []
                for image_index, image_file in enumerate(image_files):
                    outputs = augment_sem_image(
                        image_file.getvalue(), selected_methods,
                        **{**augmentation_options, "seed": int(aug_seed) + image_index},
                    )
                    for method, image_bytes in outputs:
                        preview_records.append((image_file.name, method, image_bytes))
                st.session_state.sem_augmented_previews = preview_records
                st.session_state.sem_aug_signature = current_aug_signature
                if records:
                    st.session_state.sem_augmented_labeled_records = augment_labeled_sem_records(
                        records, selected_methods, **augmentation_options
                    )
                st.success(f"已生成 {len(preview_records)} 张增强图像。")
            except Exception as exc:
                st.error(f"图像增强失败：{exc}")

        previews = st.session_state.get("sem_augmented_previews", [])
        preview_signature_matches = st.session_state.get("sem_aug_signature") == current_aug_signature
        if previews and preview_signature_matches:
            st.markdown(f"##### 增强结果预览（展示 {min(len(previews), 8)} / {len(previews)} 张）")
            preview_columns = st.columns(4)
            for index, (original_name, method, image_bytes) in enumerate(previews[:8]):
                preview_columns[index % 4].image(
                    image_bytes,
                    caption=f"{original_name} · {AUGMENTATION_METHODS[method]}",
                    use_container_width=True,
                )
            zip_buffer = BytesIO()
            with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
                for index, (original_name, method, image_bytes) in enumerate(previews):
                    original_stem = original_name.rsplit(".", 1)[0]
                    archive.writestr(f"{original_stem}__aug_{method}_{index + 1:04d}.png", image_bytes)
            st.download_button(
                "⬇️ 下载全部增强 SEM 图像（ZIP）",
                zip_buffer.getvalue(),
                "tgv_sem_augmentations.zip",
                "application/zip",
            )

        st.markdown("#### ③ 设置训练参数并开始训练")
        augmented_records = st.session_state.get("sem_augmented_labeled_records")
        can_use_augmentations = bool(records) and bool(augmented_records) and preview_signature_matches
        use_augmented_training = st.checkbox(
            "将已生成的增强图像加入训练集（验证集按原始图像分组，避免增强副本泄漏）",
            value=False,
            key="use_sem_augmented_training",
            disabled=not can_use_augmentations,
            help="请先上传标签 CSV 并生成增强图像；增强副本会与源图像分在同一训练/验证组。",
        )
        training_records = (
            augmented_records if use_augmented_training and can_use_augmentations else records
        )
        train_cols = st.columns(4)
        epochs = train_cols[0].number_input("训练轮数", 1, 200, 5, key="sem_epochs")
        batch_size = train_cols[1].selectbox("Batch size", [1, 2, 4, 8], index=0, key="sem_batch_size")
        base_channels = train_cols[2].selectbox(
            "基础通道数", [8, 16, 32, 64], index=1, key="sem_base_channels",
            help="64 为论文规模；显存有限可用 8/16，通道变化属于轻量化实现。",
        )
        learning_rate = train_cols[3].number_input(
            "学习率", min_value=1e-6, max_value=1e-2, value=1e-4, format="%.1e", key="sem_learning_rate"
        )
        can_train = bool(training_records) and torch_ok
        if st.button("🚀 开始训练 Residual U-Net", type="primary", disabled=not can_train,
                     help="需要先上传图像、标签 CSV 并通过配对检查。"):
            progress = st.progress(0, text="准备训练…")
            status = st.empty()

            def update_progress(epoch: int, train_loss: float, val_loss: float | None) -> None:
                progress.progress(epoch / int(epochs), text=f"训练中：{epoch}/{int(epochs)} 轮")
                status.write(f"Epoch {epoch}: train MSE={train_loss:.6g} · validation MSE={val_loss:.6g}")

            try:
                with st.spinner("模型训练中，请保持页面打开…"):
                    model, history = train_quality_model(
                        training_records, epochs=int(epochs), batch_size=int(batch_size),
                        learning_rate=float(learning_rate), base_channels=int(base_channels),
                        progress_callback=update_progress,
                    )
                st.session_state.sem_model = model
                st.session_state.sem_model_history = history
                st.session_state.sem_model_config = {
                    "base_channels": int(base_channels), "head_mode": "pooled", "metrics": list(QUALITY_METRICS)
                }
                st.success("训练完成。模型保存在当前浏览器会话中。")
            except Exception as exc:
                st.error(f"训练失败：{exc}")
        elif not records:
            st.caption("上传图像和标签 CSV 后，完成文件名配对，训练按钮才会启用。")

        if "sem_model_history" in st.session_state:
            history_frame = pd.DataFrame(st.session_state.sem_model_history).set_index("epoch")
            st.markdown("#### 训练损失")
            st.line_chart(history_frame)
            model = st.session_state.get("sem_model")
            if model is not None:
                buffer = BytesIO()
                import torch
                torch.save({"state_dict": model.state_dict(), **st.session_state.sem_model_config}, buffer)
                st.download_button("💾 下载训练模型权重", buffer.getvalue(), "tgv_residual_unet.pt", "application/octet-stream")

        st.markdown("#### ④ 用训练好的模型评估单张 SEM 图像")
        inference_image = st.file_uploader(
            "选择待评估 SEM 图像", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"], key="sem_inference_image"
        )
        trained_model = st.session_state.get("sem_model")
        if inference_image:
            st.image(inference_image.getvalue(), caption=inference_image.name, width=360)
            if trained_model is None:
                st.warning("请先用带专家标签的图像完成训练，再执行预测。")
            elif st.button("🔍 评估这张 SEM 图像", key="run_sem_prediction"):
                try:
                    scores = predict_sem_image(trained_model, inference_image.getvalue())
                    result_cols = st.columns(3)
                    for i, metric in enumerate(QUALITY_METRICS):
                        result_cols[i % 3].metric(metric, f"{scores[i]:.4f}")
                    current_weights = [
                        st.session_state.get(f"weight_{name}", DEFAULT_CQS_WEIGHTS[i])
                        for i, name in enumerate(QUALITY_METRICS)
                    ]
                    if abs(sum(current_weights) - 1.0) < 1e-8:
                        st.metric("预测 CQS", f"{calculate_cqs(scores, current_weights):.6f}")
                    else:
                        st.caption("CQS 权重和不为 1；请到 CQS 计算器页调整权重。")
                except Exception as exc:
                    st.error(f"图像评估失败：{exc}")

        a, b, c = st.columns(3)
        a.metric("模型输入", "SEM 图像")
        b.metric("评估输出", "NUI · SSI · TI · MCI · RI · ARI")
        c.metric("质量指标范围", "0–1")
        for name in QUALITY_METRICS:
            st.markdown(f"- **{name}**：{QUALITY_METRIC_DESCRIPTIONS[name]}")
        st.markdown("#### 网络实现模式")
        head_mode = st.radio(
            "MLP 预测头",
            ["可训练全局池化头（推荐）", "严格文献全展平头"],
            horizontal=True,
            key="unet_head_mode",
        )
        if head_mode.startswith("严格"):
            st.warning("严格头为 Linear(128×224×224, 512)，约 32.9 亿参数；仅 FP32 权重约 12.3 GiB，完整训练通常需要数十 GiB 显存/内存。")
        else:
            st.success("全局池化头保留 Residual U-Net 编码器、解码器、跳跃连接、细化层和 6 项输出，适合实际训练。")
        if torch_ok:
            st.success("PyTorch 已安装。上传配对图像/标签后可在本页启动训练和单张 SEM 推理。")
        else:
            st.error("当前环境未安装 PyTorch。请按 requirements.txt 安装后再训练；回归平台仍可正常使用。")
        st.code(
            "from laser_optimization import ResidualUNet, train_one_epoch\n"
            "model = ResidualUNet(head_mode='pooled')  # 实际训练推荐\n"
            "# criterion 默认使用 torch.nn.MSELoss()\n"
            "loss = train_one_epoch(model, real_dataloader, optimizer, device)",
            language="python",
        )
        st.caption("平台不会创建模拟图像或随机标签；DataLoader 必须由你的真实 SEM 图像和专家标注构建。")

    with cqs_tab:
        st.markdown('<div class="section-title">综合质量分数（CQS）</div>', unsafe_allow_html=True)
        st.latex(r"CQS=w_1NUI+w_2SSI+w_3TI+w_4MCI-w_5RI+w_6ARI")
        score_cols = st.columns(3)
        scores, weights = [], []
        for i, name in enumerate(QUALITY_METRICS):
            with score_cols[i % 3]:
                scores.append(st.number_input(name, 0.0, 1.0, 1.0 if name != "RI" else 0.0,
                                              0.01, key=f"cqs_{name}", help=QUALITY_METRIC_DESCRIPTIONS[name]))
                weights.append(st.number_input(f"w{i + 1}", 0.0, 1.0, float(DEFAULT_CQS_WEIGHTS[i]),
                                               0.01, format="%.6f", key=f"weight_{name}"))
        weight_sum = sum(weights)
        st.caption(f"当前权重和：{weight_sum:.6f}（必须等于 1）")
        if abs(weight_sum - 1.0) > 1e-8:
            st.error("请调整权重，使 w1 到 w6 之和等于 1。")
        else:
            cqs = calculate_cqs(scores, weights)
            st.metric("CQS", f"{cqs:.6f}")
            st.caption("RI 使用减号，因此在默认等权重下 CQS 的理论区间为 -1/6 到 5/6，并非自动映射到 0~1。")

    with bo_tab:
        st.markdown('<div class="section-title">真实实验 Ask → Tell 迭代</div>', unsafe_allow_html=True)
        st.write("填写本次真实激光参数与由 SEM 得到的六项指标，提交给高斯过程；平台再建议下一组参数。")
        st.latex(r"UCB(\mathbf p)=\mu(\mathbf p)+\kappa\sigma(\mathbf p),\quad \kappa=2.0")
        if not skopt_ok:
            st.error("当前环境未安装 scikit-optimize。安装 requirements.txt 后即可启用迭代优化。")
        if skopt_ok:
            try:
                if "laser_bo_v2" not in st.session_state:
                    st.session_state.laser_bo_v2 = IterativeBayesianOptimizer(kappa=2.0, random_state=42)
                optimizer = st.session_state.laser_bo_v2
            except Exception as exc:
                optimizer = None
                st.error(f"历史数据读取失败：{exc}")
            if optimizer is not None and optimizer.history:
                st.markdown(
                    f"#### 历史真实实验档案（{len(optimizer.history)} 条） · "
                    f"本轮 GP 已纳入 {len(optimizer.session_observations)} 条"
                )
                history_frame = pd.DataFrame(optimizer.history)
                history_frame["recorded_at"] = history_frame["recorded_at"].map(_format_history_datetime)
                st.dataframe(history_frame, hide_index=True, width="stretch")

                def history_label(index: int) -> str:
                    record = optimizer.history[index]
                    return (
                        f"第 {index + 1} 条 · {_format_history_datetime(record.get('recorded_at', '无时间'))} · "
                        f"CQS={float(record['CQS']):.5f} · 功率={float(record['laser_power_w']):g} W"
                    )

                st.selectbox(
                    "选择一条历史记录",
                    options=list(range(len(optimizer.history))),
                    format_func=history_label,
                    key="laser_history_selection",
                )
                if st.button("载入所选历史记录到本次参数表单", key="load_laser_history"):
                    record = optimizer.history[st.session_state.laser_history_selection]
                    for name in LASER_PARAMETER_NAMES:
                        st.session_state[f"bo_{name}"] = float(record[name])
                    for name in QUALITY_METRICS:
                        st.session_state[f"bo_score_{name}"] = float(record[name])
                    st.success("已载入历史参数和评分到表单；历史数据不会自动加入本轮 GP。")
            elif optimizer is not None:
                st.info("暂无历史实验记录；当前 GP 轮次为 0。记录真实结果后会自动存档。")

        param_cols = st.columns(3)
        defaults = (20.0, 1.0, 100.0, 27.0, 120.0)
        params = []
        for i, (name, low, high, label, unit) in enumerate(LASER_SPACE_SPECS):
            with param_cols[i % 3]:
                widget_key = f"bo_{name}"
                widget_options = {"min_value": low, "max_value": high, "key": widget_key}
                if widget_key not in st.session_state:
                    widget_options["value"] = defaults[i]
                params.append(st.number_input(f"{label}（{unit}）", **widget_options))
        quality_cols = st.columns(3)
        observed = []
        for i, name in enumerate(QUALITY_METRICS):
            with quality_cols[i % 3]:
                widget_key = f"bo_score_{name}"
                widget_options = {
                    "min_value": 0.0,
                    "max_value": 1.0,
                    "step": 0.01,
                    "key": widget_key,
                    "help": QUALITY_METRIC_DESCRIPTIONS[name],
                }
                if widget_key not in st.session_state:
                    widget_options["value"] = None
                observed.append(st.number_input(f"实测 {name}", **widget_options))
        if skopt_ok:
            left, right = st.columns(2)
            scores_complete = all(value is not None for value in observed)
            if not scores_complete:
                st.caption("请填写真实测量/评估得到的六项分数后再记录；空值不会被当作实验结果。")
            if left.button("记录真实结果", type="primary", width="stretch", disabled=not scores_complete or optimizer is None):
                row = optimizer.tell(params, observed)
                st.session_state.pop("bo_suggestion", None)
                st.success(
                    f"已存入历史档案第 {len(optimizer.history)} 条；本轮 GP 已纳入 "
                    f"{len(optimizer.session_observations)} 条，CQS={row['CQS']:.6f}"
                )
            if right.button("建议下一组激光参数", width="stretch", disabled=optimizer is None):
                st.session_state.bo_suggestion = optimizer.ask()
            if "bo_suggestion" in st.session_state:
                suggestion = st.session_state.bo_suggestion
                display = []
                for name, _, _, label, unit in LASER_SPACE_SPECS:
                    display.append({"参数": label, "建议值": suggestion[name], "单位": unit})
                st.dataframe(pd.DataFrame(display), hide_index=True, width="stretch")
        st.caption(
            "每次新开网页会话/重启服务后，本轮 GP 观测从 0 开始；历史档案仍可查看和载入表单，"
            "但不会自动参与当前轮拟合。持久文件：data/laser_bo_history.csv；时间按中国标准时间显示。"
        )

    with theory_tab:
        st.markdown('<div class="section-title">文献方法与本实现边界</div>', unsafe_allow_html=True)
        st.link_button(
            "📄 点击打开原始论文（IEEE Xplore）",
            "https://ieeexplore.ieee.org/document/10640121",
            type="primary",
        )
        st.markdown(
            """
            - **Residual U-Net**：四级编码器提取从孔整体形貌到微裂纹的多尺度特征，跳跃连接恢复空间细节，残差连接改善深层梯度传播。
            - **特征细化**：解码输出经过 64→256→128 两层 3×3 卷积，再由 MLP 输出 NUI、SSI、TI、MCI、RI、ARI。
            - **贝叶斯优化**：高斯过程近似昂贵的“激光参数→质量”黑箱，UCB 在利用高质量区域和探索高不确定区域之间权衡。
            - **实现边界**：PDF 正文示例末层列出 5 项指标，而任务明确要求 6 项并加入 ARI；本实现以你的 6 项输出契约为准。网页不生成模拟数据，也不把建议值冒充实验结果。
            """
        )
        st.latex(r"X_{out}=ReLU\left(X_{in}+F(X_{in},W)\right)")
        st.latex(r"k(\mathbf p,\mathbf p')=\exp\left(-\frac{\|\mathbf p-\mathbf p'\|^2}{2l^2}\right)")

