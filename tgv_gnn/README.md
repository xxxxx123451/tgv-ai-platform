# tgv_gnn

用于 TGV/TSV 铜电镀有机添加剂候选预筛的独立 P0 研究原型。该包尚未嵌入现有 Streamlit 网页，也不包含已经验证的添加剂模型。

## P0 已实现

- GAME-Net 官方 19 元素 schema 与安全元数据读取；
- PBC radius graph 和完整 PyG 数据字段；
- Cu(111)/Cu(100) slab 构建；
- GraphSAGE + Graph Multiset Transformer 基线；
- 单模型/模型集成不确定度接口与 OOD 适用域；
- 按母体分组的正确 nested CV；
- 训练折专用目标标准化、早停和最佳权重恢复；
- 全局与家族回归指标；
- 分角色、分晶面的筛选规则；
- POSCAR + sidecar JSON 导出与 DFT 标签回填；
- 版本化 YAML、JSON Schema 和测试。

## 安装

建议使用独立虚拟环境，避免影响当前网页：

```powershell
cd tgv_gnn
python -m pip install -e ".[research,test]"
```

## 验证

```powershell
python -m pytest tests -q
python scripts/p0_smoke.py
```

## 重要边界

P0 的模拟 fixture 只验证软件，不产生科研结论。真正的候选排序至少需要统一 DFT 协议的 Cu(111) 标签、盲测评估、OOD/不确定度报告和 Top-K DFT 复核。未知元素不会被映射成全零向量；未标定阈值时筛选器只排序，不执行硬过滤。

Streamlit Community Cloud 未来只承担轻量推理、数据检查和结果展示。大规模 GNN 训练、吸附构型生成和 VASP 必须在本地或计算服务器执行。
云端轻量推理只需安装核心依赖；RDKit 和 ASE 分别位于 `chemistry` / `dft`
可选依赖中，避免在推理容器中无条件安装重型工具。
