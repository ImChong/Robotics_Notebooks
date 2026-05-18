# HoloMotion-1 Technical Report（arXiv:2605.15336）

> 来源归档（ingest）

- **标题：** HoloMotion-1 Technical Report
- **缩写：** **HoloMotion-1**
- **类型：** paper / humanoid motion foundation model / zero-shot whole-body motion tracking
- **arXiv：** <https://arxiv.org/abs/2605.15336>（HTML：<https://arxiv.org/html/2605.15336v1>）
- **PDF：** <https://arxiv.org/pdf/2605.15336>
- **作者：** Maiyue Chen, Kaihui Wang, Bo Zhang, Xihan Ma, Zhiyuan Yang, Yi Ren, Qijun Huang, Zihao Zhu, Yucheng Wang, Zhizhong Su（Horizon Robotics）
- **入库日期：** 2026-05-18
- **一句话说明：** 面向 **零样本全身运动跟踪** 的人形 **运动基础模型**：以 **野外视频重建动作为主、MoCap 与自采动作为辅** 的混合语料规模化训练；策略为 **稀疏 MoE Transformer + KV-cache 推理** 与 **序列级 PPO**，在未见数据集上泛化并在 **无任务特化微调** 下迁移真机。

## 摘要级要点（与 abs 一致）

- **数据范式：** **大规模混合运动语料**——**野外视频重建运动**提供主要 **多样性**，**精选 MoCap + 室内自采** 提供 **高保真监督与部署向覆盖**；相对纯 MoCap 训练显著拓宽行为、视角与风格覆盖，但引入重建噪声、域失配与质量不均。
- **建模：** **大容量时序建模**；**稀疏激活 MoE Transformer**，推理端 **KV-cache** 以满足 **实时闭环** 延迟；**序列级训练策略** 提升长片段上的学习效率（论文报告相对基线在效率与跟踪误差上的大幅改进，具体数字以原文为准）。
- **任务表述：** 目标条件 **POMDP**；观测含本体感知与带 **短 horizon 前瞻** 的参考运动特征；奖励为 **稠密跟踪 + 稳定性/正则** 组合；低层为 **归一化关节目标 + PD 力矩** 接口。
- **实验：** 多个 **未见** 运动基准、与既有方法对比、效率分析、**直接真机迁移**（无任务特化微调）。

## 官方工程与分发（与仓库正文一致）

- 代码：<https://github.com/HorizonRobotics/HoloMotion>
- 文档站：<https://horizonrobotics.github.io/robot_lab/holomotion>（Horizon Robotics GitHub Pages 路径，与社区 `fan-ziqi/robot_lab` 扩展库 **同名不同主体**）
- 权重：<https://huggingface.co/HorizonRobotics/HoloMotion_models>
- Docker：<https://hub.docker.com/r/horizonrobotics/holomotion>

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/holomotion.md`](../../wiki/entities/holomotion.md)
- 互链参考：[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BFM](../../wiki/entities/paper-behavior-foundation-model-humanoid.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md)、[AMASS](../../wiki/entities/amass.md)、[模仿学习 / PPO](../../wiki/methods/imitation-learning.md)
