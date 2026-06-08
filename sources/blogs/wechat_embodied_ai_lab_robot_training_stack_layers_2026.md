# 有了Isaac Lab和MuJoCo，为什么还会出现 mjlab、Newton、UniLab、Genesis？

> 来源归档（blog / 微信公众号）

- **标题：** 有了Isaac Lab和MuJoCo，为什么还会出现 mjlab、Newton、UniLab、Genesis？
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/Z9pgVa48wQKLYVRD3psnhw
- **发表日期：** 2026-06-08（frontmatter）
- **入库日期：** 2026-06-08
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（修复 hatchling `force-include` 重复后 `pip install -e` + `agent-reach install --channels=wechat`）；微信正文经 `~/.agent-reach/tools/wechat-article-for-ai`（Camoufox）；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **姊妹篇：** [42 篇 humanoid RL 身体系统栈](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)（`hz9JXtJeUPRfUGzfD-pZuA`）；[机器人世界模型训练闭环](wechat_embodied_ai_lab_robot_world_model_training_loop.md)（`0edW0GhwtyNc5nF6RDIfuw`）
- **一句话说明：** 克制地重框「仿真框架变天」叙事：Isaac Lab / MuJoCo 仍是主流牌桌，mjlab、UniLab、Newton、Genesis World 等是在**不同层级**补厚训练–评估闭环；核心判断是机器人学习最贵之处正从**单次仿真速度**转向**整条闭环返工成本**，训练栈正从训练工具演变为覆盖物理、渲染、运行时、评估与真机闭环的基础设施。

## 核心摘录（归纳，非全文）

### 主判断

- **没有洗牌，但在分层：** 大平台（Isaac Lab）、物理/sim2sim（MuJoCo/MJX/Warp）、任务入口（Playground/mjlab）、异构运行时（UniLab）、底层连接器（Newton）、闭环评估基础设施（Genesis World）**不在同一层竞争**。
- **成本重心转移：** 环境搭得快不快、奖励改完多久见效、sim2sim/真机能否复用、失败样本能否回流——比峰值 FPS 更决定迭代速度。
- **Genesis 补全评估维：** 仿真作为机器人基础模型的**评估与迭代引擎**（非仅数据生成器）；博客叙事含 **200+ h 真机评估 → 0.5 h 仿真** 与 **89% 相关性**（以官方材料为准）。

### 六层分工（文章归纳）

| 层 | 代表 | 回答的问题 |
|----|------|------------|
| 大平台 | Isaac Lab / Isaac Sim | 多模态传感、USD 资产、大规模并行、复杂场景统一工作台 |
| 物理 / sim2sim | MuJoCo、MJX、MuJoCo Warp | 透明动力学、快速验证、接触调试 |
| 任务与训练入口 | MuJoCo Playground、mjlab | **time-to-robot**：从行为想法到真机可见策略的墙钟 |
| 运行时 / 异构调度 | UniLab | CPU 批量物理 + GPU 学习；采集–学习重叠；跨 macOS/ROCm/XPU |
| 底层物理引擎 | Newton | OpenUSD + Warp + MuJoCo Warp 连接器；可微与多求解器 |
| 闭环评估基础设施 | Genesis World | 渲染 + 统一多物理 + Quadrants 编译器 + real-to-sim 评测 |

### 各块要点

| 项目 | 文章强调 |
|------|----------|
| **Isaac Lab** | 非 Isaac Gym 换皮；OpenUSD / PhysX / Lab Views 统一场景–物理–学习接口；重但适合 Loco-Manip / VLA / 多传感 |
| **MuJoCo Playground** | MJX 上多类机器人环境；压短训练→部署；降低复现摩擦 |
| **mjlab** | manager-based 模块化（obs/reward/event/command/curriculum）接到 MuJoCo Warp；长期任务维护 |
| **UniLab** | 3–10× 端到端墙钟（论文）；采样器–学习器时序消融；非 CUDA 路径 |
| **Newton** | Isaac Lab 3.0 Beta develop 集成仍在快速开发；PhysX↔Newton 策略迁移与 G1 部署叙事 |
| **Genesis World** | Simulation Interface / Nyx Render / 统一 Physics / Quadrants Compiler 四层栈 |

### 与人形 / 具身栈的关系

- 运动控制：奖励、先验、地形课程、本体参数的快速比较依赖训练栈厚度。
- Loco-Manip / VLA / 世界模型：需要闭环评估、后训练与数据回流接口。
- 与 [身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) 互补：该文偏**工具链分层**，姊妹篇偏**算法/身体能力分层**。

## 对 wiki 的映射

- [robot-training-stack-layers-technology-map](../../wiki/overview/robot-training-stack-layers-technology-map.md)（本次升格主页面）
- [isaac-lab](../../wiki/entities/isaac-lab.md)、[mujoco](../../wiki/entities/mujoco.md)、[mujoco-playground](../../wiki/entities/mujoco-playground.md)、[mjlab](../../wiki/entities/mjlab.md)、[unilab](../../wiki/entities/unilab.md)、[newton-physics](../../wiki/entities/newton-physics.md)、[genesis-world-10](../../wiki/entities/genesis-world-10.md)
- [simulator-selection-guide](../../wiki/queries/simulator-selection-guide.md)、[simulation-evaluation-infrastructure](../../wiki/concepts/simulation-evaluation-infrastructure.md)、[mujoco-vs-isaac-lab](../../wiki/comparisons/mujoco-vs-isaac-lab.md)

## 可信度与使用边界

- 本文为**策展解读**，各项目细节以官方文档 / 论文 / 仓库 README 为准。
- Newton、Genesis 等快速演进条目须核对克隆日版本；文中性能与相关性数字引用自作者转述的官方叙事。
- 微信 CDN 图片未纳入 wiki 正文。

## 当前提炼状态

- [x] Agent Reach + Camoufox 正文抓取与六层归纳
- [x] 与既有实体页交叉映射确认
- [x] 主 overview 页升格
