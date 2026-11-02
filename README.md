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

`D1, D2, i0, c_ACC, c_SUP, c_LEVE, Area_ratio`

页面会训练 ANN、XGBOOST、RF、GPR、SVR、PR 六种模型，显示 R²、RMSE、MAE，并支持单样本预测、保存模型和 Optuna（ANN/XGBOOST）搜索。

## 代码调用

```python
from ml_models import train_models, save_bundle
models, metrics, x_test, y_test = train_models(df)
print(metrics)
save_bundle(models, "tgv_models.joblib")
```

PR 使用五阶 `PolynomialFeatures`；ANN 使用 `(100, 50)`、ReLU、`learning_rate_init=0.0005`、`batch_size=64`；XGBOOST `max_depth=10`；RF `max_depth=15`；GPR 使用 `ConstantKernel * RBF`；SVR 使用 RBF 和 `gamma="scale"`。
