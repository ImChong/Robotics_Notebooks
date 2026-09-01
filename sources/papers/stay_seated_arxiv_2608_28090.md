# Stay Seated（G1 坐姿全向移动）

> 来源归档（ingest）

- **标题：** Stay Seated: Learning Omnidirectional Humanoid Locomotion on a Passive Mobile Chair with Casters
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2608.28090>
- **机构：** 大阪大学（Osaka University）系统创新系；东京大学神经智能国际研究中心（Horii）
- **平台：** Unitree G1（29 DoF）+ 五万向轮被动移动椅
- **仿真：** [mjlab](https://github.com/mujocolab/mjlab)
- **入库日期：** 2026-09-01
- **一句话说明：** 在标准站立速度跟踪环境上最小扩展（被动椅模型、坐姿奖励、非对称 actor–critic），无运动模仿即可学习 G1 坐姿全向速度跟踪，并零样本 sim2real。

## 核心摘录（策展）

### 1) 问题：坐姿 loco-manipulation 的第一步

- **摘录要点：** QDD 人形站立需持续关节力矩发热；人类桌边工作常坐椅.delegate 承重。本文作为 **seated loco-manipulation** 第一步，研究**被动万向椅**上全向 $(v_x,v_y,\omega_z)$ 跟踪：骨盆–椅面接触不固定，靠脚–地间歇推进整体。
- **对 wiki 的映射：**
  - [Stay Seated](../../wiki/entities/paper-stay-seated.md) — 任务定义。
  - [Humanoid Locomotion](../../wiki/tasks/humanoid-locomotion.md) — 非常规接触拓扑。

### 2) 方法：站立环境 + 椅模型 + 非对称 AC

- **摘录要点：** Actor 仅本体感知 + 速度指令（96 维），**无**脚/椅接触传感；Critic 特权观测含椅相对位姿、万向轮与骨盆–椅接触（217 维）。奖励含任务跟踪、躯干/椅直立、坐姿保持、脚滑惩罚等。骨盆碰撞网格细化 + 椭圆摩擦锥抑制穿透与侧滑。
- **对 wiki 的映射：**
  - [Stay Seated](../../wiki/entities/paper-stay-seated.md) — 观测与奖励。
  - [Unitree G1](../../wiki/entities/unitree-g1.md) — 硬件平台。

### 3) $2^3$ 因子实验：SY / FS / CC

- **摘录要点：** 四种子 × 八条件（对称正则 SY、脚滑惩罚 FS、指令课程 CC）。随机指令评估：八条件 timeout 成功率均 ≥99.45%；**SY+CC** 平移跟踪 RMSE 最低。FS 单独使用可降 CoT 但增跟踪误差，且易收敛**静止局部最优**；与 SY 或 CC 组合可避免。方向解析：CoT 排序 backward < lateral ≪ forward。
- **对 wiki 的映射：**
  - [Stay Seated](../../wiki/entities/paper-stay-seated.md) — 消融读法。

### 4) Sim2Real

- **摘录要点：** 学习策略（仅 actor 观测）**零样本**部署真机 G1，实现全向坐姿移动（论文 Fig.1 / 视频叙事）。
- **对 wiki 的映射：**
  - [Sim2Real](../../wiki/concepts/sim2real.md) — 非对称 critic 特权训练范式。

### 5) 开源状态（截至 2026-09-01）

- **摘录要点：** arXiv 与摘要**未列出**官方代码 / 项目页 / 权重；复现依赖公开 **mjlab** + Unitree G1 资产，环境改动需自行实现。标为 **未开源（方法论文）**。
- **对 wiki 的映射：**
  - [Stay Seated](../../wiki/entities/paper-stay-seated.md) — 局限节。

## 当前提炼状态

- [x] arXiv 摘要与方法节已对齐摘录
- [x] 无项目页：按论文口径记录未开源
- [x] wiki 映射：`wiki/entities/paper-stay-seated.md`
