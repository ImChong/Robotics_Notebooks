# 没有遥操作也能训人形VLA？斯坦福 LEGS 用 3DGS 降低 loco-manip 数据成本

> 来源归档（blog / 微信公众号）

- **标题：** 没有遥操作也能训人形VLA？loco-manip时代来了！斯坦福 LEGS 用 3DGS 把 loco-manip 数据成本打下来
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/B1sYOPKg6TQwnNGs-_8NDw
- **发表日期：** 2026-06-04
- **入库日期：** 2026-06-04
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0（editable 安装 + `wechat-article-for-ai`/Camoufox）；正文约 0.4 万字 / 12 图；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **关联姊妹篇：** [42 篇 humanoid RL 身体系统栈](wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)、[Ego 9 篇专题](wechat_embodied_ai_lab_ego_9_papers_survey.md)、[BFM 41 篇专题](wechat_embodied_ai_lab_bfm_41_papers_survey.md)
- **一句话说明：** 策展解读斯坦福 **LEGS**（arXiv:2606.01458）：用 **3DGS 背景 + SAM3D 前景 mesh + MuJoCo 物理** 合成可复用人形 loco-manip 演示，**无遥操作** 微调 ψ0 / π0.5 / GR00T N1.6，在 Unitree G1 上匹配或超过 50 条 teleop 基线，并强调 **motion–appearance 解耦重渲染** 的数据工厂逻辑。

## 核心论文（单篇）

| 字段 | 内容 |
|------|------|
| 论文 | LEGS: Fine-Tuning Teleop-Free VLAs for Humanoid Loco-manipulation in an Embodied Gaussian Splatting World |
| 机构 | Stanford University |
| arXiv | https://arxiv.org/abs/2606.01458 |
| 项目页 | https://legsvla.github.io/ |

## 核心摘录（归纳，非全文）

### 问题重框

- **双重要求：** 合成数据须 **物理可执行**（MuJoCo + 接触）且 **视觉接近真机头摄**（否则 VLA 在域差上失效）。
- **数据瓶颈：** 人形 loco-manip 的 teleop 贵、难复用；换场景/物体/提示词往往要重新采。

### 3DGS 的角色（策展观点）

- **不是装饰：** 3DGS 背景 + 两阶段颜色校准，把合成图拉到部署相机分布；mesh-only（SAM3D）平均成功率约 **33%**，LEGS(200) 约 **67%**（文内引用论文 Table 2 叙事）。
- **复用核心：** motion 与 appearance 解耦 → 同一段 18-D 命令流可在新背景/新物体上 **GPU 重渲染**（Task 3：新条件 teleop **>1.5 h** vs LEGS **~0.1 h**）。

### 管线要点

| 模块 | 作用 |
|------|------|
| 视觉前端 | 手持视频 → 3DGS 背景；物体图 → SAM3D mesh；深度合成 + 颜色校准 |
| 物理后端 | MuJoCo 500 Hz；SONIC 低层 WBC；18-D 高层命令（双臂腕 SE(3)+夹爪 + 底座 4-D） |
| 程序化生成 | Walk / Pick / Place 等 primitive 组合；仅保存验证通过的 episode |
| VLA 微调 | ψ0、π0.5、GR00T N1.6 公开 checkpoint 上微调 |

### 三个任务与关键结果（文内强调）

| Task | 难度 | Teleop(50) Task 3 | LEGS(200) Task 3（示例） |
|------|------|-------------------|-------------------------|
| 1 | 桌面 pick-place | — | 全 backbone 上 LEGS 最好或并列 |
| 2 | 走到桌边再操作 | — | 同上 |
| 3 | 走–拿–转身–蹲放 | **0/10**（三 backbone） | 5/10、2/10、6/10 |

### 边界（文内保守判断）

- 依赖已有 **SONIC** 低层控制器；motion 来自 **程序化 primitive**，非轨迹优化最优；
- 新场景仍需 **1–2 分钟** 手持视频建 3DGS；评测以 **G1 + D435** 为主；
- 动态背景、透明/反光物体、跨人形硬件泛化未充分验证。

## 对 wiki 的映射

- [paper-legs-embodied-gaussian-splatting-vla](../../wiki/entities/paper-legs-embodied-gaussian-splatting-vla.md)（论文实体 + Mermaid 管线）
- [legs_arxiv_2606_01458.md](../papers/legs_arxiv_2606_01458.md)（arXiv 摘录）
- 交叉：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[VLA](../../wiki/methods/vla.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[GS-Playground](../../wiki/entities/gs-playground.md)、[VIRAL](../../wiki/entities/paper-viral-humanoid-visual-sim2real.md)

## 可信度与使用边界

- 本文为 **微信公众号策展导读**；定量表格与消融以 [arXiv:2606.01458](https://arxiv.org/abs/2606.01458) 与项目页为准。
- 原始抓取正文见 [wechat_legs_vla_3dgs_2026-06-04.md](../raw/wechat_legs_vla_3dgs_2026-06-04.md)。
