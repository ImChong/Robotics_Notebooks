# Robocurve — robocurve.org

- **类型：** 公司站点 / 独立物理 AI 评测机构（原始资料归档）
- **收录日期：** 2026-09-06
- **主链接：** <https://robocurve.org/>
- **GitHub org：** <https://github.com/robocurve>（旗舰仓 [inspect-robots](https://github.com/robocurve/inspect-robots)）
- **文档：** <https://docs.inspectrobots.org/>
- **背书：** Y Combinator（首页 Backed by Y Combinator）

## 一句话

**Robocurve** 是 **Public Benefit Corporation（公益公司）**，主张对 **physical AI / 机器人能力** 做 **独立、开放、可复现** 的 **真机优先** 评测，并向公众发布能力报告；旗舰开源工具为 **Inspect Robots**（MIT）。

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| **Inspect Robots 框架** | **已开源** — [robocurve/inspect-robots](https://github.com/robocurve/inspect-robots)，MIT，~298 stars（2026-09-06） |
| **评测任务目录 WorldEvals** | **已开源** — README 指向 [robocurve/worldevals](https://github.com/robocurve/worldevals) |
| **各机型插件** | 分仓 MIT（如 `inspect-robots-yam`、`inspect-robots-franka` 等） |
| **被测 VLA/LLM 权重** | **不在本 org 内** — 由 OpenPI、XPolicyLab、各模型 API 等第三方提供 |
| **能力报告数据** | 站点图表与报告为 Robocurve 发布；复现需按报告说明 + Inspect Robots 日志 |

## 首页叙事（2026-09-06 抓取）

### 定位

- **Real-world evaluations of physical AI** — 测量并向公众报告前沿机器人能力。
- **独立评测机构：** 反对 cherry-picked demo；提供连续、标准化的 robotics 能力测量。
- **公益使命：** PBC，依法服务公共利益；认为通用人形机器人可能在数年内落地，对经济与劳动市场影响深远。

### 公开报告主题（站点展示）

| 主题 | 要点 |
|------|------|
| Opus 5 & Tower of Hanoi | 长程任务能力案例 |
| Test-time scaling | LLM **thinking effort / API cost** 与 robotics score 正相关 |
| LLM inference speed | 模型推理速度随时间指数改善（多模型曲线） |
| Opus 5 block stacking | 积木堆叠等 manipulation 案例 |
| Voice-guided instruction | 语音引导指令跟随 |

### 开源工具（Featured）

**Inspect Robots** — 「If you know Inspect AI, this is that for robotics.」

- 真机优先，支持仿真
- 可跑 **VLA、WAM、LLM、coding agents**
- 集成 **ROS、Isaac Lab、Cap-X、XPolicyLab**
- **MIT** 许可；完整 trace log + **Rerun** 可视化

### 合作与联系

- Featured by / With support from experts（站点列出 MATS、OpenAI、DeepMind、Princeton、MIT 等专家 testimonial）
- Contact：Evaluation / Funding / Collaboration

## 对 wiki 的映射

- [robocurve](../../wiki/entities/robocurve.md) — 机构实体
- [inspect-robots](../../wiki/entities/inspect-robots.md) — 评测框架实体
- [hub-embodied-eval-benchmark](../../wiki/overview/hub-embodied-eval-benchmark.md)
- [xpolicylab](../../wiki/entities/xpolicylab.md) — 策略 serving 插件集成
- [isaac-lab-arena](../../wiki/entities/isaac-lab-arena.md) — Isaac Lab 仿真 embodiment 插件
