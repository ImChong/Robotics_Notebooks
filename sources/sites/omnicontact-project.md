# OmniContact 项目页（omnicontact.github.io）

> 来源归档

- **标题：** OmniContact: Chaining Meta-Skills via Contact Flow
- **类型：** site（项目页 + 交互 demo + 数据集可视化）
- **URL：** <https://omnicontact.github.io/>
- **论文：** <https://arxiv.org/abs/2606.26201>
- **机构：** 诺亦腾机器人（Noitom Robotics）、香港科技大学（HKUST）、武汉大学（WHU）、香港大学（HKU）
- **入库日期：** 2026-07-01
- **一句话说明：** 官方页：框架概览、MuJoCo WASM + ONNX 在线策略 viewer、OmniContact 数据集 t-SNE/动作预览、meta-skill 泛化/chaining/recovery 视频、VLM 语义任务与 G1 交互 GLB 序列、baseline 对比。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| Overview | CF-Track + CF-Gen 分层框架；Carry Box **98.7%**、Push Suitcase **82.5%**、~**40 min** 耐力、**50 Hz** 闭环监控 |
| Live MuJoCo Session | 浏览器内 MuJoCo WASM + ONNX policy viewer（全页入口） |
| Dataset Visualization | 74,641 MoCap clips t-SNE；1,274 序列 / 22.29 h / 7.22M 物体帧 / 90 Hz 同步 |
| Generalized Meta Skill | carry / push / kick / slide / relocate；高度/形状/初末位姿泛化 |
| Meta Skill Chaining | push-stack、stack-boxes 等组合任务 |
| Recovery | 掉箱/推箱失败后的自主恢复演示 |
| VLM Integration | 自然语言→物体布局（心形、进球、N-O-I-T-O-M 字母等） |
| Baseline Comparison | vs carry/push/stack/push-stack 基线视频 |
| Interactive G1 Motion | GLB 交互 viewer（Unitree G1 运动序列） |

## 对 wiki 的映射

- 主实体：[OmniContact（论文实体）](../../wiki/entities/paper-omnicontact-humanoid-loco-manipulation.md)
- 论文摘录：[omnicontact_arxiv_2606_26201.md](../papers/omnicontact_arxiv_2606_26201.md)
- 代码：[omnicontact-sim2sim.md](../repos/omnicontact-sim2sim.md)
- 数据集：[omnicontact-dataset.md](../datasets/omnicontact-dataset.md)
