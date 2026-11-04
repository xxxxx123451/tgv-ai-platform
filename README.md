---
title: TGV Electroplating And Laser AI
emoji: 🔬
colorFrom: purple
colorTo: pink
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# TGV 电镀填充六模型回归平台

## P0 算法全谱系平台

依据《ML算法全谱系规格说明书_TSV_TGV_电子电镀.md》新增 `configs/`、`src/`、`scripts/`、`tests/` P0 工程。它严格采用 TSV/TGV 数据契约字段和单位，派生 AR、添加剂比、有效电流密度、扩散长度与 Peclet-like 特征；所有数值预处理均在 Pipeline 内完成，支持 RepeatedKFold、RepeatedStratifiedKFold、GroupKFold 和 TimeSeriesSplit。

快速运行合成数据基准：

```bash
python scripts/run_benchmark.py --target thickness_um --models ridge svr rf xgboost lightgbm catboost --out runs/t4
pytest -q
```

输出 `runs/t4/ranking.md`、`ranking.csv` 和 `model_cards.json`。LightGBM/CatBoost 未安装时，基准会将对应模型标记为 skipped 而不影响其他模型。

P0 模型卡包含 task、algorithm、data、best_params、metrics、baseline_comparison、uncertainty、interpretability、limitations、seed 和 versions 字段。厚度与空洞面积默认使用 log1p 目标变换；分类模型接口支持验证集 F1 阈值搜索。

## 规格说明书中的文献基线

1. Ding X. et al., *Electrochimica Acta* 557, 148450 (2026), DOI: 10.1016/j.electacta.2026.148450：TSV 铜填充 ANN/XGBoost/RF/GPR/SVR/Polynomial 对比。
2. Ouyang Y. et al., *IEEE TCPMT* 14, 1680 (2024), DOI: 10.1109/TCPMT.2024.10640121：TGV 激光参数、Residual U-Net 与贝叶斯优化。
3. Katırcı R., Oğuz T., *Adv. Mater. Sci. Eng.* (2025), DOI: 10.1155/amse/6234549：Ni–Cr 电镀 RF 与 GA 优化。
4. Huang C., Cohen J., *ASME MSEC* 2025, MSEC2025-155351：电镀浓度虚拟量测与不确定性。
5. Ngaruiya A. M., GVSU MSc Thesis (2025)：电沉积 ANN/PINN/Hybrid PINN 加速仿真。
6. Fernández et al., *Coatings* 14, 979 (2024)：电镀涂层 XGBoost/RF 与 SHAP/PDP。
7. Ofir N. et al., arXiv:2202.02998：无监督异常学习的 SEM 缺陷分割。

## 安装与运行

```bash
python -m venv .venv
.venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
```

上传 CSV/Excel，表头必须包含：

`Depth, Wide, i0, c_ACC, c_SUP, c_LEVE, Area_ratio`

页面会训练 ANN、XGBOOST、RF、GPR、SVR、PR 六种模型，显示 R²、RMSE、MAE，并支持单样本预测、保存模型和 Optuna（ANN/XGBOOST）搜索。

## 代码调用

```python
from ml_models import train_models, save_bundle
models, metrics, x_test, y_test = train_models(df)
print(metrics)
save_bundle(models, "tgv_models.joblib")
```

PR 使用五阶 `PolynomialFeatures`；ANN 使用 `(100, 50)`、ReLU、`learning_rate_init=0.0005`、`batch_size=64`；XGBOOST `max_depth=10`；RF `max_depth=15`；GPR 使用 `ConstantKernel * RBF`；SVR 使用 RBF 和 `gamma="scale"`。

## 论文风格 Parity Plot

“训练与评估”页面完成训练后自动显示独立测试集真实值与预测值对比图。
可选择全部已训练模型或单个模型；六模型排列为 ANN/XGBOOST/RF、GPR/SVR/PR。
点击“生成投稿图片”后下载 600 dpi PNG 和矢量 PDF，也可直接下载测试预测 CSV。
为避免内存累积，网页只保留当前选项的投稿导出；新训练或导入模型会清除旧评估。
仅导入 joblib 不包含测试集，需使用原始实验数据重新训练以生成评估图。

```python
from parity_plot import plot_parity
import matplotlib.pyplot as plt

# y_true: 一维测试集真实值；predictions: {"ANN": 一维预测值, ...}
fig = plot_parity(y_true, predictions, save_path="parity_plots.png")
plt.show()
plt.close(fig)
```

或者对网页下载的 CSV 运行：

```bash
python parity_plot.py --csv test_predictions.csv --output parity_plots.png
```

样式参考 Ding et al. 的 Figure 10（DOI: 10.1016/j.electacta.2026.148450）。
原图为 3×2，此实现按需求改为 2×3；网页仅计算 Area_ratio 指标，
不复用论文的五目标平均指标。图上保留 `RMAE` 标签，但其数值为标准 RMSE。
不将超出 0～1 的输入或预测裁剪，以免掩盖数据或模型问题。

## ANN SHAP 参数重要性

当侧栏明确选择“单模型训练 → ANN”时，训练结束后还会自动计算并显示
ANN 的 SHAP Summary Beeswarm Plot。解释使用本轮测试样本和训练集背景；Depth 与 Wide
作为一个联合特征组参与联盟枚举，并以 `Aspect ratio = Depth / Wide` 显示。
ANN 的六列训练输入不变，多模型对比模式不会显示该图。
页面可下载 600 dpi PNG、矢量 PDF 和逐样本 SHAP 数值 CSV。独立绘图脚本：

```python
from shap_beeswarm import plot_shap_beeswarm
fig = plot_shap_beeswarm(
    shap_values, feature_values, feature_names,
    output_name="Area_ratio", save_path="shap_beeswarm.png",
)
```

直接运行 `python shap_beeswarm.py` 会生成 Figure 12 方向的演示图；模拟值仅用于
检查排版，不会进入网页训练或模型解释。
