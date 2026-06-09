# COINS（ETH Zürich / Google 人–场景交互语义合成）

> 来源归档

- **标题：** COINS — Compositional Human-Scene Interaction Synthesis with Semantic Control
- **类型：** repo
- **来源：** Kaifeng Zhao（zkf1997）/ ETH Zürich & Google
- **链接：** <https://github.com/zkf1997/COINS>
- **论文：** <https://arxiv.org/abs/2207.12824>
- **项目页：** <https://zkf1997.github.io/COINS/index.html>
- **入库日期：** 2026-06-09
- **一句话说明：** ECCV 2022 **COINS** 官方实现：**PelvisVAE + BodyVAE** 两阶段 Transformer cVAE、**PROX-S** 数据扩展、**组合交互推理**（`--composition 1`），以及 **PiGraph-X / POSA-I** 基线与评测脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-coins-compositional-human-scene-interaction.md`](../../wiki/entities/paper-coins-compositional-human-scene-interaction.md)

---

## 核心定位

**COINS** 从 **3D 场景 + 语义 action–object 规格** 生成 **SMPL-X 人体** 与场景的自然交互。仓库除论文方法外，还发布 **PROX-S**（PROX 上的实例分割与逐帧交互语义标注）及预训练 checkpoint。

---

## 环境与依赖（README 摘要）

| 组件 | 版本/说明 |
|------|-----------|
| Python | 3.7 |
| PyTorch | 1.11.0 + CUDA 11.3 |
| PyTorch3D | 0.6.2 |
| 训练框架 | PyTorch Lightning |
| 人体模型 | SMPL-X（需自行下载权重） |
| 网格下采样 | POSA `mesh_ds` |

---

## 主要目录与入口

| 路径 | 用途 |
|------|------|
| `interaction/two_stage_sample.py` | **推理主入口**：`--interaction 'sit on-chair[+touch-table]'`，`--composition 1` 启用组合模式 |
| `interaction/interaction_trainer.py` | **BodyVAE** 训练 |
| `interaction/transform_trainer.py` | **PelvisVAE** 训练 |
| `data/load_interaction.py` | 按 action / action–object 加载 PROX-S 交互并可视化 |
| `data/scene.py` | 渲染场景分割、记录物体实例 |
| `pigraph/` | **PiGraph-X** 基线 |
| `POSA/` | **POSA-I** 放置管线（需合并上游 POSA） |
| `evaluation/` | 物理可行性、语义接触、多样性评测 |

---

## 采样示例（README）

```bash
cd interaction
# 原子交互
python two_stage_sample.py --interaction 'sit on-chair' --scene_name 'MPH16' --visualize 1
# 复合交互（训练数据含复合时）
python two_stage_sample.py --interaction 'sit on-chair+touch-table' --scene_name 'MPH16' --visualize 1
# 组合模式（仅原子数据训练）
python two_stage_sample.py --interaction 'sit on-chair+touch-table' --scene_name 'MPH16' \
  --composition 1 --transform_checkpoint 'pelvis_atomic.ckpt' --interaction_checkpoint 'body_atomic.ckpt'
```

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [CRISP](../papers/crisp_real2sim_iclr2026.md) | 同在 **PROX** 人–场景生态；COINS **前向生成** 填充场景，CRISP **后向 Real2Sim** 从视频恢复可仿真资产 |
| [TokenHSI](../papers/bfm_awesome_tokenhsi_arxiv_2503_19901.md) | 均建模 **人–场景交互语义**；COINS 面向 **静态姿态合成**，TokenHSI 面向 **人形控制 task token** |

## 对 wiki 的映射

- 实体页：[COINS（论文）](../../wiki/entities/paper-coins-compositional-human-scene-interaction.md)
