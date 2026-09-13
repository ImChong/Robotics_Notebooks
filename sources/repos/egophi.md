# EgoPHI

> 来源归档

- **标题：** EgoPHI — Estimating 3D Hand-Object Contact and Force from Egocentric Vision
- **类型：** repo（训练 / 评测 / 力仿真 / 预处理）
- **机构：** 苏黎世联邦理工（ETH Zürich）；SIPLAB（eth-siplab）
- **链接：** <https://github.com/eth-siplab/EgoPHI>
- **项目页：** <https://siplab.org/projects/EgoPHI>
- **论文：** <https://arxiv.org/abs/2608.13014>
- **Hugging Face：** <https://huggingface.co/datasets/eth-siplab/EgoPHI>
- **入库日期：** 2026-09-13
- **许可证：** MIT（README）
- **代码 / 开源状态：** **部分开源** — 训练/评测/预处理/`force_sim` **已发布**；HF 力标注与真机数据 **已发布**；README 预训练权重链接 **为空**（需 `train.py` 自训）
- **一句话说明：** EgoPHI 官方实现：InteractionGNN 三阶段管线 + SOFA 力监督生成 + ARCTIC/H2O 评测与真机数据集发布入口。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-egophi.md`](../../wiki/entities/paper-egophi.md)
- **交叉归档：** [egophi-siplab.md](../sites/egophi-siplab.md)、[egophi_arxiv_2608_13014.md](../papers/egophi_arxiv_2608_13014.md)

---

## 仓内结构（README 摘要）

| 路径 | 作用 |
|------|------|
| `model.py` | `InteractionGNN` 定义（Graph-Based Interaction Blocks） |
| `train.py` | ARCTIC 训练入口 |
| `evaluate_ARCTIC.py` / `evaluate_H2O.py` | 域内 / 跨数据集评测 |
| `compute_metrics.py` | 接触与力指标汇总 |
| `arctic_preprocess.py` / `h2o_preprocess.py` | 数据预处理全流程 |
| `force_sim/` | SOFA 物理仿真生成力监督 |
| `config.py` | 路径与超参（支持环境变量覆盖） |
| `inspect_predictions.ipynb` | 单帧预测可视化 |

## 最短复现路径

```bash
conda env create -f environment.yml && conda activate egophi
# clone HACO_RELEASE 邻仓；下载 ARCTIC + H2O
# HF: arctic_force_simulations.zip / h2o_force_simulations.zip
python arctic_preprocess.py
python train.py                    # 或等待官方 checkpoint
python evaluate_ARCTIC.py
python evaluate_H2O.py
python compute_metrics.py --predictions-dir ... --gt-dir ...
```

- 评测可用 `EGOPHI_EVAL_MAX_SAMPLES` 做 smoke test。
- 真机数据：`egophi_dataset.zip` on Hugging Face。

---

## 对 wiki 的映射

- 实体页：[EgoPHI](../../wiki/entities/paper-egophi.md)
- 方法交叉：[模仿学习](../../wiki/methods/imitation-learning.md)、[WiLoR](../../wiki/methods/wilor.md)
- 同实验室对照：[EgoExoMoCap](../../sources/repos/egoexomocap.md)（另一 SIPLAB ego 项目，代码待发布）
