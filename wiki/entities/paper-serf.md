---
type: entity
project_id: serf
tags:
  - paper
  - vla
  - mobile-manipulation
  - memory
  - ucsd
  - umich
status: complete
updated: 2026-11-12
arxiv: "2606.12956"
original: ../../sources/papers/serf_arxiv_2606_12956.md
related:
  - ../methods/vla.md
  - ../methods/imitation-learning.md
  - ../methods/behavior-cloning.md
  - ../entities/behavior-1k.md
  - ../entities/paper-pi05-open-world-vla.md
  - ../entities/paper-pi05-inference-settings-behavior-1k.md
  - ../concepts/flow-matching-embodied-policy.md
  - ../concepts/lora.md
sources:
  - ../../sources/papers/serf_arxiv_2606_12956.md
summary: "SERF（arXiv:2606.12956）：用共享潜空间里的 neural points 同时表示环境与机器人本体，在线更新后切成 8 个 map token 喂给 π0.5 系 VLA；在 BEHAVIOR-1K 三个长程移动操作任务（仅仿真）上任务进度均值约 44.0% 升至 58.7%。"
---

# SERF（arXiv:2606.12956）

**SERF**（*SERF: Spatiotemporal Environment and Robot Feature Map for Long-Horizon Mobile Manipulation*，[arXiv:2606.12956](https://arxiv.org/abs/2606.12956)，[项目页](https://existentialrobotics.org/serf/)）：UC San Diego 等的工作，作者 Sunghwan Kim、Byeonghyun Pak 等。入选 CoRL 2026（按 Glen Berseth 的阅读清单标注为 Spotlight，会场信息未在论文本身核实）。

## 一句话定义

**把环境和机器人本体都表示成同一潜空间里的 neural points 特征图，随交互在线更新，再把地图切成 8 个 token 作为 VLA（π0.5）的额外状态输入，用于长程移动操作。**

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SERF | Spatiotemporal Environment and Robot Feature map | 本文的时空环境与机器人特征图 |
| VLA | Vision-Language-Action | 视觉–语言–动作策略，本文基座为 π0.5 |
| VFM | Vision Foundation Model | 视觉基础模型，本文用 DINOv3 提供 patch 嵌入 |
| FK | Forward Kinematics | 前向运动学，用于摆放机器人点 |
| FGR | Fast Global Registration | 快速全局配准，SE(3) 估计的初值 |
| ICP | Iterative Closest Point | 迭代最近点，SE(3) 精修 |
| BDDL | Behavior Domain Definition Language | BEHAVIOR 的任务目标描述语言，本文用于过滤任务相关物体与计算子目标进度 |
| OOD | Out-of-Distribution | 分布外，测试时改变布局 |
| LoRA | Low-Rank Adaptation | 低秩适配，本文只在动作专家里插入 |

## 为什么重要

- 作者的问题（§1）：长程移动操作要持续回答「我在哪、环境变了什么、任务做到哪」，而多数 VLA 没有显式机制保留当前观测之外的信息。
- 与既有路线的差别（作者自述，§1）：静态 3D 特征图方法不随交互更新，也不把机器人本体放进地图；SERF 同时补上这两点。
- 对读者：它是「显式空间记忆 + VLA」的一个具体实现，且给出了与静态地图基线（SBP）和纯图像基线的同设置对照（§5）。

## 核心原理与流程总览

```mermaid
flowchart LR
  A[RGB-D 与实例标签] --> B[DINOv3 patch 嵌入 反投影到 3D]
  B --> C[离线学习 neural points 潜特征与共享解码器]
  C --> D[环境点 对象级 SE3 更新]
  C --> E[机器人点 前向运动学更新]
  D --> F[Point Transformer 分 8 头出 map token]
  E --> F
  F --> G[map token 加图像 状态 任务嵌入]
  G --> H[π0.5 动作专家 流匹配出动作块]
```

- **地图（§3，附录 A/B）：** 点集 P = {位置, 潜特征, 实例标签}。查询位置 x 时取 K 个近邻，按 softmax(-距离/σ) 插值潜特征，再经共享 MLP 解码器还原 DINOv3 嵌入。重建损失为 1 − 余弦相似度；另有类别间与实例内（部件级）对比损失（λ_inter=0.02，λ_intra=0.01，附录 C）。附录 B 的数值：DINOv3 1280 维、480×480 输入、30×30 patch 网格、体素 0.02 m、每点 64 维特征、K=6、温度 0.05。
- **更新（§3，附录 D）：** 只改点坐标，特征、标签、解码器固定。环境按实例当作刚体，用 CoTracker3 跟踪 2D 关键点并提升到 3D，FGR 初始化加 ICP 精修；实例质心位移小于 0.015 m 视为静止跳过。机器人点由 URDF 采样、按前向运动学定位。
- **map token（§4，附录 E）：** 先按 BDDL 只保留机器人点与任务相关物体点，采样 25000 点，经共享两级 Point Transformer 后接 8 个分支头：基座系 1 m / 2 m / 4 m 球查询共 3 个，末端左右 0.5 m 共 2 个，另有仅机器人、仅环境、全局各 1 个。每个 token 2048 维，与 VLA token 维度一致。为减少过拟合，token 特征不含绝对坐标。
- **策略训练（§4，附录 E）：** 基于 π0.5，冻结 VLM 与视觉编码器，动作专家插入 LoRA；条件流匹配目标；动作块长 30，保留前 26 步做三次插值重采样为 20 条控制指令；每任务微调 20k 步，批大小 16。

## 工程实践

- **基座与对照设置（§5）：** PI0.5 (pre) 是 BEHAVIOR-1K 全 50 任务微调的检查点（取自 Larchenko 等的 OpenPI 实现）；其余变体都从它初始化，同一训练设置，只在 map token 输入上不同。SBP 与 SERF (env) 没有机器人点，用 6 个 token。
- **数据（附录 G）：** 每任务 200 条专家示范，20 个评测配置，最大回合长度设为平均示范长度的 2 倍。三个任务：21 收集儿童玩具（共 7 个目标，4.4×3.0 m）、22 鞋放鞋架（5.5×9.0 m）、26 组装礼品篮（7.6×8.8 m）。
- **特权信息（§5 实现细节）：** 为简化数据关联，使用仿真给出的特权实例标签，而非从 RGB 推出。
- **记录的计算量：** 论文未报告地图构建、跟踪或推理延迟，复现成本无法从文中判断。论文未给出代码仓库。

## 评测证据与清单说法的对比核对

清单一句话摘要：「Conditions a VLA on an online spatiotemporal neural-point map of the scene and the robot body for long-horizon mobile manipulation.」（作者说明是 LLM 据摘要起草，排序为兴趣打分而非质量评价。）

| 清单措辞 | 论文核对 | 结论 |
|----------|----------|------|
| conditions a VLA | §4：以 π0.5 为基座，map token 作为额外输入 token，冻结 VLM 加 LoRA 动作专家 | 吻合 |
| online spatiotemporal neural-point map | §3：neural points，环境点按对象刚体 SE(3)、机器人点按前向运动学在线更新。但「在线」更新的只有点坐标；潜特征与解码器是执行前离线学好的先验地图（局限节） | 基本吻合，限定词需补充：依赖先验地图 |
| of the scene and the robot body | 环境点与机器人点共享解码器与潜空间（§3） | 吻合 |
| long-horizon mobile manipulation | 评测为 BEHAVIOR-1K 的 3 个仿真任务；清单未提「仅仿真」「依赖仿真特权实例标签」。作者在局限节明确说无真机实验、与图像基线的比较并非同等输入信号 | 清单遗漏了仿真与特权信息这两条关键限定 |

## 结论

SERF 展示了持久空间记忆接入 VLA 的收益，但现有结果依赖先验地图与仿真特权标签。

- 比较视觉基线时明确额外地图和实例信息，避免把不对等输入当纯架构增益。
- 分别测试地图更新、机器人本体表示与 token 切分，并按任务报告进度。
- 真机、铰接和可变形物体需重新验证，三个仿真任务的部分得分不能当实际完成率。

## 局限

作者自述（局限节）：

1. 依赖执行前学好的先验地图和仿真特权实例标签，因此与纯图像基线的比较是「在这些假设下加入持久空间记忆的收益」，不是输入完全对等的比较。
2. 没有真机实验，向真实世界迁移未验证。
3. 时间更新只支持对象级刚体运动，铰接与可变形物体不在范围内。
4. 只用单一时间窗口的 map token，未纳入过去观测或预测未来状态。
5. map tokenizer 的空间子集是手工指定的，没有把任务级语义与机器人 / 物体 / 场景状态分开。

笔者阅读时看到的未验证之处（判断，非作者所述）：

- 效果幅度：Table 1 中任务 26 上 SERF 52.5±13.8 与 SBP 51.6±11.7、PI0.5 (ft) 48.4±21.1 的差距小于各自标准差；SERF (env) 在任务 26 为 49.4±13.4，低于 SBP。三个任务的均值提升主要来自任务 21 和 22。
- 标准差很大（多在 12 到 24 之间），每配置只有 20 个，论文未给显著性检验。
- 失败恢复实验的掉落是人为在运输中张开夹爪造成，且仅对比 PI0.5 (ft) 一个基线，每组 20 次。
- 地图 token 消融（附录 F，Table 2）只在任务 22 上做，且每个变体只列均值与标准差，未与其余任务交叉验证。
- 与 SBP 的比较是否为公平的容量对比（SERF 多了机器人点和更新逻辑）：论文称所有变体共享基座与训练设置，但参数量或 token 数差异（6 对 8）只在 §5 一句话里交代。

## 关联页面

- [VLA（Vision-Language-Action）](../methods/vla.md)
- [模仿学习](../methods/imitation-learning.md)
- [行为克隆](../methods/behavior-cloning.md)
- [BEHAVIOR-1K](../entities/behavior-1k.md)
- [π0.5：开放世界 VLA](../entities/paper-pi05-open-world-vla.md)
- [π0.5 推理设置与 BEHAVIOR-1K](../entities/paper-pi05-inference-settings-behavior-1k.md)
- [具身策略中的流匹配](../concepts/flow-matching-embodied-policy.md)
- [LoRA](../concepts/lora.md)

## 参考来源

- [serf_arxiv_2606_12956.md](../../sources/papers/serf_arxiv_2606_12956.md)
- [arXiv:2606.12956](https://arxiv.org/abs/2606.12956)
- [项目页](https://existentialrobotics.org/serf/)

## 继续阅读

- [arXiv PDF](https://arxiv.org/pdf/2606.12956)
- [VLA 方法页](../methods/vla.md)
- [BEHAVIOR-1K](../entities/behavior-1k.md)
