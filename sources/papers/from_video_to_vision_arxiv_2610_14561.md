# From Video to Vision: Learning Generalizable Humanoid Carrying from Human Motion（MOVER，arXiv:2610.14561）

> 来源归档（ingest）

- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2610.14561>（v1，2026-10-10）
- **HTML：** <https://arxiv.org/html/2610.14561v1>
- **作者：** Ziluo Ding
- **机构：** 北京人形机器人创新中心（X-Humanoid）
- **项目 / 代码：** 截至 2026-10-24，arXiv 页面未列独立项目页或可访问代码仓库。
- **入库日期：** 2026-10-24
- **一句话说明：** 将人类搬运视频重定向为人形参考动作，通过对象与接触增强、教师–助教–学生式分阶段 RL 和 QuatNet，把固定轨迹模仿逐步转为依靠机载 RGB-D 的人形搬运。
- **对 wiki 的映射：** [MOVER 论文实体页](../../wiki/entities/paper-mover-humanoid-carrying.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

## 核心论文摘录（MVP）

### 1) 视频数据与交互一致的动作扩增（§III-A）

- **链接：** <https://arxiv.org/html/2610.14561v1>
- **摘录要点：** 用 GENMO 从搬运视频恢复世界坐标人体动作，再用 GMR 重定向到人形。作者保持示范中的手–物关系，扰动物体尺寸、形状、初始位姿和支撑高度，并通过受约束逆运动学调整手臂与躯干，生成覆盖不同物体配置的接触一致参考轨迹。
- **对 wiki 的映射：** [MOVER 论文实体页](../../wiki/entities/paper-mover-humanoid-carrying.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md)

### 2) 三阶段策略从跟踪转向位置泛化（§III-B）

- **链接：** <https://arxiv.org/html/2610.14561v1>
- **摘录要点：** Stage 1 以 PPO 学习严格跟踪全身和物体参考轨迹的教师；Stage 2 移除全局锚点位置与自身体位置跟踪奖励、强化物体与接触奖励，得到能适应不同物体位置的助教；Stage 3 用 DAgger-PPO 在减少位置特权观测后蒸馏学生，并继续用 PPO 优化。
- **对 wiki 的映射：** [MOVER 论文实体页](../../wiki/entities/paper-mover-humanoid-carrying.md)、[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 3) 免参考动作与视觉部署（§III-C–D、§IV-E）

- **链接：** <https://arxiv.org/html/2610.14561v1>
- **摘录要点：** 为进一步移除部署时的参考动作观测，作者以 LSTM 版 QuatNet 从物体相对位姿、关节位置和目标点回归人形锚点参考朝向。机载 RGB-D 相机检测物体上的标记；短时遮挡时结合手部正运动学更新物体位姿。天工 3.0 真机在三个放置位置的视觉策略成功率分别为 7/10、5/10、5/10。
- **对 wiki 的映射：** [MOVER 论文实体页](../../wiki/entities/paper-mover-humanoid-carrying.md)、[X-Humanoid](../../wiki/entities/x-humanoid.md)

## 来源与身份说明

本文为 arXiv:2610.14561，MOVER 是论文中提出的搬运学习框架名称。来源仅支持论文预印本；截至入库日，未找到该论文列出的独立项目页或代码仓库，因此不将相关工具 GENMO、GMR 或机器人平台仓库误记为本文实现。
