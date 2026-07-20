# 刚刚，机器人顶会 RSS 三项最佳论文出炉！708 篇送审，仅 8 篇杀入决赛

> 来源归档（blog / 微信公众号）

- **标题：** 刚刚，机器人顶会RSS三项最佳论文出炉！708篇送审，仅8篇杀入决赛
- **类型：** blog
- **作者：** 量子位（微信公众号 QbitAI）
- **原始链接：** https://mp.weixin.qq.com/s/M3gYuB1gB2c3XL1GMk-p-A
- **发表日期：** 2026-07-16
- **入库日期：** 2026-07-20
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对该链接触发微信 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_qbitai_rss2026_awards_2026-07-16/`](../raw/wechat_qbitai_rss2026_awards_2026-07-16/)
- **一句话说明：** 报道 **RSS 2026** 论文奖：投稿 708 / 录用 210（29.7%）；**Final List 8 篇** 中评出最佳论文 / 最佳学生论文 / 最佳系统论文；并覆盖 IJRR & RSS **Test of Time**、Early Career Spotlight。本库对奖项与 Final List 论文建/链独立节点（已有节点不重复造页）。

## 核心摘录（归纳，非全文）

### 奖项与录用

| 指标 | 数值 |
|------|------|
| 投稿 | 708 |
| 录用 | 210（接收率约 29.7%） |
| Final List | 8（先从 Strong Accept 候选筛至半决赛再至决赛） |

### 三项大奖 + Final List（8）

| 角色 | 论文 | 链接 | wiki |
|------|------|------|------|
| **最佳论文** | FlashSAC | [arXiv:2604.04539](https://arxiv.org/abs/2604.04539) | [flashsac](../../wiki/methods/flashsac.md)（已有，本次交叉更新） |
| **最佳学生论文** | Muninn: Your Trajectory Diffusion Model But Faster | [arXiv:2605.09999](https://arxiv.org/abs/2605.09999) | [paper-muninn-…](../../wiki/entities/paper-muninn-trajectory-diffusion-acceleration.md)（已有） |
| **最佳系统论文** | NeuralActuator | [arXiv:2607.11734](https://arxiv.org/abs/2607.11734) | [paper-neuralactuator-…](../../wiki/entities/paper-neuralactuator-neural-actuation-modeling.md)（已有） |
| Finalist | Automated Synthesis of Facial Mechanisms… | [arXiv:2607.11688](https://arxiv.org/abs/2607.11688) | [paper-automated-facial-mechanisms-animatronic](../../wiki/entities/paper-automated-facial-mechanisms-animatronic.md) |
| Finalist | OAT: Ordered Action Tokenization | [arXiv:2602.04215](https://arxiv.org/abs/2602.04215) | [paper-oat-ordered-action-tokenization](../../wiki/entities/paper-oat-ordered-action-tokenization.md) |
| Finalist | Emerging Extrinsic Dexterity…（DAPL） | [arXiv:2603.09882](https://arxiv.org/abs/2603.09882) | [paper-dapl-extrinsic-dexterity-clutter](../../wiki/entities/paper-dapl-extrinsic-dexterity-clutter.md) |
| Finalist | cuNRTO | [arXiv:2603.02642](https://arxiv.org/abs/2603.02642) | [paper-cunrto-gpu-robust-trajectory-optimization](../../wiki/entities/paper-cunrto-gpu-robust-trajectory-optimization.md) |
| Finalist | Realizing Robotic Swimming… Multiphysics | [arXiv:2506.05012](https://arxiv.org/abs/2506.05012) | [paper-unified-fluid-robot-multiphysics-swimming](../../wiki/entities/paper-unified-fluid-robot-multiphysics-swimming.md) |

### 时间检验奖

| 奖项 | 论文 | wiki |
|------|------|------|
| **IJRR Test of Time** | The EuRoC Micro Aerial Vehicle Datasets | [euroc-mav-datasets](../../wiki/entities/euroc-mav-datasets.md) |
| **RSS Test of Time** | Compliant, Underactuated Robotic Hand（Deimel & Brock, RSS 2014） | [paper-deimel-compliant-underactuated-robotic-hand](../../wiki/entities/paper-deimel-compliant-underactuated-robotic-hand.md) |

### Early Career Spotlight（文内四人，未单独建学者页）

李弘扬（HKU / Whole-Body Intelligence）、Marco Tognon（Inria，空中物理交互）、Pulkit Agrawal（MIT，力智能与终身学习）、Wenzhen Yuan（UIUC，触觉智能）。本 ingest **不新建人物 stub**；相关主题已由现有 [VLA](../../wiki/methods/vla.md)、[sim2real](../../wiki/concepts/sim2real.md)、[manipulation](../../wiki/tasks/manipulation.md) 等枢纽承接。

## 对 wiki 的映射

- **新建 5 篇 Finalist + 2 篇 ToT** 实体页（见上表）；**三项大奖** 复用已有完整节点并回链本博客。
- 交叉：`flashsac` / `diffusion-policy` / `actuator-network` / `vla` / `manipulation` / `trajectory-optimization` / `sim2real` 等。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] Final List / 大奖 / ToT 全部独立节点或复用已有完整页
- [x] 项目页开源状态核查（OAT / Facial / Aquarium.jl / DAPL in-prep / cuNRTO 未列仓等）
