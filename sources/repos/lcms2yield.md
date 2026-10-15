# xtalpi-inc/lcms2yield

- **标题：** lcms2yield — AI-Predicted Full-Spectrum UV Correction Factors
- **类型：** repo
- **来源：** XtalPi
- **链接：** <https://github.com/xtalpi-inc/lcms2yield>
- **许可证：** Apache-2.0（仓库 README）
- **配套论文：** [JACS Au 2026, DOI 10.1021/jacsau.6c00812](https://doi.org/10.1021/jacsau.6c00812) — [论文摘录](../papers/lcms2yield_jacsau_2026.md)
- **官方介绍：** [XtalPi 研究介绍](../sites/lcms2yield.md)
- **入库日期：** 2026-10-15

## 一句话说明

公开用于预测全光谱 UV 校正因子（CF）的代码和数据，支持研究论文提出的无分析物标准品浓度与产率定量方法。

## 仓库内容与入口

| 路径 / 入口 | 用途 |
| --- | --- |
| `model/predict.py` | README 示例使用的 SMILES 列表 CF 预测接口 `predict_by_smiles_list` |
| `model/model_pth/` | 模型文件目录 |
| `data/quantum_chemistry/data.csv` | 22,609 个量子化学预训练化合物 |
| `data/experiment/Fine_tuning.csv` | 1,845 个 LC-UV 实验微调化合物 |
| `data/experiment/FDA.csv` | 144 个 FDA 药物浓度独立测试集 |
| `data/experiment/Reaction.csv` | 178 个真实反应产率测试样本 |
| `data/experiment/Isolated_Yields_Suzuki_34.csv` | Suzuki 反应分离收率对照数据 |
| `data/experiment/Isolated_Yields_Minisci_26.csv` | Minisci 反应分离收率对照数据 |

**运行环境（README）：** Python 3.11；列出 PyTorch、PyTorch Geometric、RDKit、Performer 等依赖，并提供 CPU 安装说明；NVIDIA GPU/CUDA 用于可选加速。按仓库 README 安装后可从 SMILES 调用预测接口。

## 开源边界

- **已开源：** 推理代码、模型目录与多组训练/评测数据均在仓库公开。
- README 示例接口输出预测 CF；从仪器读取 LC-UV 原始信号、峰识别和整套实验自动化控制不属于该示例接口所展示的范围。

## 与 wiki 的关系

- 实体页：[paper-lcms2yield](../../wiki/entities/paper-lcms2yield.md)
