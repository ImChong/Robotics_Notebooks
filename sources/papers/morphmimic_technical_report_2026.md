# MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids

> 来源归档（ingest）

- **标题：** MorphMimic: Morphology-Conditioned World Model and Policy Sharing Across Humanoids
- **类型：** paper（technical report）
- **报告：** <https://lzyang2000.github.io/morphmimic/static/morphmimic.pdf>
- **项目页：** <https://lzyang2000.github.io/morphmimic/>（[来源归档](../sites/morphmimic-project.md)）
- **代码：** <https://github.com/lzyang2000/morphmimic>（[来源归档](../repos/morphmimic.md)）
- **作者：** Lizhi Yang、Fengze Xie、Yuyang Wang、Lu Gan、Yisong Yue、Aaron D. Ames
- **机构：** 加州理工学院（Caltech）、佐治亚理工学院（Georgia Tech）
- **年份：** 2026
- **入库日期：** 2026-10-25
- **一句话说明：** 在 SuperMimic 的可微世界模型动作跟踪路线之上，MorphMimic 将统一机器人表示、形态条件化与跨机器人潜变量对齐结合，使 G1、X2 Ultra、T2 共用世界模型和动作跟踪策略。

## 核心摘录（技术报告）

- **基线关系：** 报告称 MorphMimic 建立在 SuperMimic 思路上，但其 SuperMimic 基础算法是作者从论文独立实现的，并非 SuperMimic 作者代码，也未获其审阅或背书。
- **共同接口：** 按关节运动学功能映射到 G1 的 29 关节规范顺序，并将机器人映射到 14 个跟踪刚体槽位；每个本体另带关节符号、零位偏移和动作映射。
- **形态条件化与共享：** 125 维形态描述符包含执行器参数、关节增益/力矩限制、肢段几何和质量。策略解码器使用 FiLM 调制，世界模型通过形态描述符条件化。网络权重共享，但归一化量、参考动作、回放缓存和机器人常量按机器人保留。对应参考动作的隐变量通过对齐损失拉向跨机器人均值。
- **共享世界模型和策略：** 训练一个可微世界模型及一个完整动作跟踪策略，策略在冻结的世界模型上做 32 步想象 rollout；控制频率为 50 Hz。
- **两种指令接口：** 全身动作参考；五点稀疏目标（骨盆、左右腕、左右踝），由独立的五点编码器接入同一策略解码器，可用于 VR 遥操作。
- **仿真评测：** 在三个机器人训练动作库共有的 128,786 段片段上，各共享配置在六种机器人/指令组合上都与单机器人专家策略的平均回报差距不超过 0.02。统一策略的片段生存率为 99.729%–99.891%；此指标指确定性仿真 rollout 到达动作片段末帧的比例，不等同于未见动作泛化或真机成功率。
- **真机边界：** 统一策略在 Unitree G1 真机上展示舞蹈、行走和实时遥操作；报告把 X2 Ultra、T2 真机验证列为未来工作。
- **算力读法：** 统一训练用三张 H100 NVL 共 16.3 GPU-hours；三个单机器人专家合计 13.9 GPU-hours。因此核心贡献是跨本体共享和统一接口，不是低于独立专家总算力。

## 对 wiki 的映射

- [MorphMimic：跨人形机器人共享可微世界模型与跟踪策略](../../wiki/entities/paper-morphmimic.md)
- [SuperMimic：论文与研究实体](../../wiki/entities/paper-supermimic.md)
