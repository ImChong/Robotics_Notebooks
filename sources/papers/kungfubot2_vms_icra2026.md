# KungfuBot 2: Learning Versatile Motion Skills for Humanoid Whole-Body Control

- **标题：** KungfuBot 2: Learning Versatile Motion Skills for Humanoid Whole-Body Control
- **类型：** paper
- **会议：** ICRA 2026
- **arXiv：** <https://arxiv.org/abs/2509.16638>
- **项目页：** <https://kungfubot2-humanoid.github.io/>
- **代码：** <https://github.com/TeleHuman/PBHC>（2025-10 起 general motion tracking 分支）
- **机构：** TeleAI、SJTU、ECUST
- **收录日期：** 2026-07-09
- **一句话说明：** 提出 **VMS** 统一全身控制器：单策略掌握多样动态技能并保持 **分钟级长序列** 稳定；**混合局部/全局跟踪目标** + **正交混合专家 OMoE** + **段级跟踪奖励**，仿真与真机均优于 ExBody2/GMT，并可作 text-to-motion 低层控制器。

## 核心摘录（策展，非全文）

### 1）问题

- 单策略需覆盖 **广泛动作库**，同时在 **长时域** 保持稳定性——比 v1 单动作/小集合更难。
- 逐步匹配（step-wise）在急转、侧踢等场景易失衡；纯全局跟踪在快速跑动急转时累积漂移。

### 2）VMS 框架

| 组件 | 作用 |
|------|------|
| **Hybrid tracking objective** | 平衡 **局部 motion 保真** 与 **全局轨迹一致性**，缓解根速度跟踪带来的漂移 |
| **OMoE（Orthogonal Mixture-of-Experts）** | 鼓励技能子空间专业化，同时提升跨动作泛化；专家用法随动作复杂度自适应（走路少专家、舞蹈多专家） |
| **Segment-level reward** | 放松逐步刚性匹配，用短未来窗口提升对全局位移与瞬时误差的鲁棒性 |

### 3）实验亮点

- 主结果：VMS 在局部/全局误差上优于 ExBody2、GMT；长序列（按时长分组）基线随序列变长退化，VMS 保持稳健。
- 真机：风格化步态（走/正步/跑）、羽毛球/抛球/挥拍、多样踢腿、武术（组合拳、马步、李小龙 pose）、长序列舞蹈与武术（太极、少林拳、七星拳等）。
- **下游：** text-to-motion 自然语言指令跟踪；少量微调可适应空翻等 OOD 极限技能。

### 4）与 v1 / PBHC 关系

- 同一 TeleHuman 团队与 PBHC 代码库；v2 算法以 **general motion tracking** 形式合入仓库（2025-10 news）。
- v1 侧重 **高动态单技能 + 自适应容差课程**；v2 侧重 **单策略多技能 + 长序列 + OMoE**。

## 对 wiki 的映射

- [paper-notebook-kungfubot-2](../../wiki/entities/paper-notebook-kungfubot-2.md) — 主实体页
- [pbhc.md](../repos/pbhc.md) — 统一训练/部署栈
- [paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont](../../wiki/entities/paper-notebook-kungfubot-physics-based-humanoid-whole-body-cont.md) — v1 对照
- [imitation-learning](../../wiki/methods/imitation-learning.md) — 混合跟踪 + 段级奖励语境
- [whole-body-control](../../wiki/concepts/whole-body-control.md) — 通用低层 WBC 底座

## 参考来源（原始）

- 项目页：<https://kungfubot2-humanoid.github.io/>
- 论文：<https://arxiv.org/abs/2509.16638>
- 代码：<https://github.com/TeleHuman/PBHC>
