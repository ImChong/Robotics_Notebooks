# TacPAC（arXiv:2609.05266）

> 来源归档（ingest）

- **标题：** TacPAC: Tactile Prediction and Real-Time Action Correction in World-Action Models for Contact-Rich Manipulation
- **简称：** TacPAC
- **类型：** paper / world-action-model / tactile / contact-rich-manipulation
- **arXiv：** <https://arxiv.org/abs/2609.05266>
- **PDF：** <https://arxiv.org/pdf/2609.05266>
- **代码：** <https://github.com/LogosRoboticsGroup/TacPAC> — 归档见 [`sources/repos/logos-robotics-tacpac.md`](../repos/logos-robotics-tacpac.md)
- **机构：** 复旦大学数据科学学院、上海创智学院（SII）、NeoteAI
- **入库日期：** 2026-09-08
- **一句话说明：** 把 WAM 的触觉预测缓存成 layer-wise KV，执行期用触觉专家对照缓存做实时动作修正；五任务真机平均 22%→64%，单次修正 30.4 ms（20.7× 快于重生成 chunk）。

## 开源状态（步骤 2.5，2026-09-08）

| 组件 | 状态 |
|------|------|
| GitHub | **已开源** MIT；含模型、训练、预处理、部署与单测 |
| 数据集 / 检查点 | README 写明 **正在准备公开发布** |

**结论：部分开源** — 代码与训练/推理栈可复现；权重与数据集待发布。

## 核心摘录

### 摘录 1：预测对齐的实时修正

- 纯视觉 WAM 缺局部接触线索；把未来触觉当额外视角预测只能拿到约 **1/3** 可达增益。
- 时序错配：预测在执行前固定，触觉在执行中到达。
- TacPAC 在 base chunk 规划后缓存「该计划预期的接触 + 动作表征」；每帧新触觉与缓存对照，只改 **尚未执行** 的后缀。
- 单次修正 **30.4 ms（32.9 Hz）**，比整 chunk 重生成 **20.7×** 更快。

**对 wiki 的映射：** [paper-tacpac](../../wiki/entities/paper-tacpac.md)

### 摘录 2：两阶段训练与真机任务

- Stage 1：视频专家预测未来视觉+触觉，动作专家通过 MoT 联合去噪 action chunk。
- Stage 2：冻结 base；采样执行偏移，监督未执行后缀的 delta action。
- 五类接触丰富任务（精密插入、易碎物、重定向、长程等），每任务 20 次真机试验；TacPAC **五任务全胜**，平均 **64%** vs 纯视觉 base **22%**。

**对 wiki 的映射：** [paper-tacpac](../../wiki/entities/paper-tacpac.md)

## 当前提炼状态

- [x] 仓库与 README 核查（2026-09-08）
- [x] wiki 映射：`wiki/entities/paper-tacpac.md`
