---
type: entity
tags: [robotics, embodied-ai, benchmark, manipulation, coding-agent, physical-evaluation]
status: active
updated: 2026-10-21
project_id: aa-robotics
project: https://artificialanalysis.ai/articles/aa-robotics
related:
  - ./robodojo.md
  - ./paper-hexashell.md
  - ../tasks/manipulation.md
  - ../concepts/sim-vs-real-eval-gap.md
  - ../queries/embodied-eval-benchmark-selection-loop.md
sources:
  - ../../sources/sites/artificial-analysis-aa-robotics.md
summary: "Artificial Analysis 的 AA-Robotics v0.1：用 coding-agent Harness 让五个前沿语言模型通过 HTTP API 零样本操控 Hiwonder NexArm，完成 8 项桌面任务；首批 GPT-6 Astra 得分 69.4/100，但每项平均约 30 分钟，显著慢于人工 20 秒。"
---

# AA-Robotics v0.1

**AA-Robotics v0.1** 是 Artificial Analysis 于 2026-10-14 发布的具身智能研究预览：让通用语言模型在 coding-agent Harness 中直接操控真机械臂，测量其理解、规划和物理执行能力。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| AA | Artificial Analysis | 独立 AI 模型与系统评测机构；此处指其机器人评测 |
| API | Application Programming Interface | 模型通过简单 HTTP API 与机械臂交互 |
| LLM | Large Language Model | 评测对象；在 coding-agent Harness 中使用 |
| HTTP | Hypertext Transfer Protocol | 控制容器与机械臂之间的接口协议 |
| Zero-shot | Zero-shot evaluation | 任务执行前不针对该任务进行示例训练 |

## 评测设置

模型运行在隔离容器，得到简短的机械臂型号、HTTP 控制 API 与任务完成提交方式说明；可读取桌面三台相机与机械臂腕部相机画面，并在容器内编写、运行代码来规划和发送动作。硬件为 Hiwonder NexArm，文章仅注明 6 轴（5 关节与夹爪），没有提供更多规格。GPT 系列通过 Codex，Claude 系列通过 Claude Code，Gemini 4 Argon 通过 Antigravity，均设置为 high effort。

## 八项桌面任务

| 任务 | 主要压力点 |
|------|------------|
| 按下红色按钮 | 接触定位与动作控制 |
| 将鳄鱼玩具放进矩形框 | 目标定位、抓取与放置 |
| 把饮料倒入杯中 | 稳定抓取、倾倒与液体控制 |
| 将积木收进盒子并盖好 | 多物体操作、顺序和遮挡 |
| 完成形状匹配 | 视觉识别、空间关系与精细放置 |
| 写下 hello world | 工具控制与轨迹精度 |
| 搭最高的塔 | 稳定性、接触与长时规划 |
| 测量盒子长宽高 | 几何推理与测量精度 |

任务完成得 100 分，部分完成依进度得分；报告的总榜分为跨任务平均分。

## 首批结果

| 模型 | AA-Robotics 分数 | 单项平均耗时 |
|------|------------------:|---------------|
| GPT-6 Astra | 69.4（正文约为 69） | 约 30 分钟 |
| GPT-6.1 Sol | 64.6 | 约 30 分钟 |
| Gemini 4 Argon | 55.3 | 约 52 分钟 |
| Claude Opus 5.5 | 48.8 | 约 30 分钟 |
| Claude Fable 5.1 | 46.3 | 约 30 分钟 |

模型整体平均每个任务约半小时，人工平均约 20 秒，相差约 100 倍。Astra 平均每项成本为 16.36 美元。该基线让能力与效率差距同时可见：分数已显示若干简单任务可完成，但运行时延和成本还远离生产使用。

## 如何解读

- **速度不是独立于能力的维度。** 长时间观察和推理、反复试探与恢复都会拉长任务耗时；效率优化不能只缩短模型思考，也需减少无效动作。
- **抓取只是失败链的一环。** 原文错误记录包含空抓、夹持后滑落、错位放置，以及接触时推走其他物体等问题。
- **判断与控制耦合。** 识别几何关系、预测碰撞、判断是否完成、根据反馈改变策略共同决定最终得分。八项任务覆盖面有限，不能直接外推到移动操作、人机协作或开放环境。
- **分数应与设置一起读。** 这是一个机械臂、八个桌面任务、各家 coding-agent 产品与 high-effort 配置下的研究预览，不是通用机器人能力的完整排名。
- **开源与复现：** 截至 2026-10-21，官方文章未列公开代码仓库或可复现实验 Harness；评测 API 与容器细节也未完全披露。

## 与相关基准的关系

AA-Robotics 是以语言模型 coding agent 为控制主体的真机桌面操作评测。与 [RoboDojo](./robodojo.md) 的仿真/真机策略基准、[HexaShell](./paper-hexashell.md) 的轻量 Agent Harness 研究有关联，但任务集合、硬件、执行接口和评分口径不同，数值不可直接横向比较。在 [具身大模型评测基准选型闭环](../queries/embodied-eval-benchmark-selection-loop.md) 中，它属于 ③ 策略任务成功率评测层的真机桌面操作样本，适合与仿真基准一起做 sim↔real 校准。

## 参考来源

- [官方文章归档](../../sources/sites/artificial-analysis-aa-robotics.md)
- [Artificial Analysis 原文](https://artificialanalysis.ai/articles/aa-robotics)
- [Manipulation](../tasks/manipulation.md)
- [仿真 vs 真机评测 gap](../concepts/sim-vs-real-eval-gap.md)
