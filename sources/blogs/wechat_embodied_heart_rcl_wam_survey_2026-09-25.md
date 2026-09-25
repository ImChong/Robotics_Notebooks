# 近300篇工作调研！WAM 模型的训练策略：数据、预训练与后训练的种种。

> 来源归档（blog / 微信公众号）

- **标题：** 近300篇工作调研！WAM 模型的训练策略：数据、预训练与后训练的种种。
- **类型：** blog / wechat / survey / wam
- **作者：** 具身智能之心（编辑；原文作者 Zuxing Lu 等）
- **原始链接：** https://mp.weixin.qq.com/s/0ZMRkQCTmbWxWDiUBvdbsg
- **发表日期：** 2026-09-25（推断；配套 arXiv:2609.16074 中文导读）
- **入库日期：** 2026-09-25
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`wechat_embodied_heart_rcl_wam_survey_2026-09-25.md`](../raw/wechat_embodied_heart_rcl_wam_survey_2026-09-25.md)
- **配套论文：** [RCL WAM 综述](../papers/rcl_wam_robot_learning_survey.md) · **arXiv:[2609.16074](https://arxiv.org/abs/2609.16074)**
- **一句话说明：** 对 MBZUAI/RCL *World-Action Models for Robot Learning and Control: A Survey* 的中文导读——WM/VLA/WAM 分界、2×2 架构、三类数据金字塔、预训练/后训练闭环与开放挑战；同步综述 **arXiv 编号** 并交叉既有 [Awesome WAM（RCL）](../../wiki/entities/awesome-world-action-models-rcl.md) 与 [WAM 概念页](../../wiki/concepts/world-action-models.md)。

## 核心摘录（归纳，非全文）

### 总判断

- 文内「近 300 篇」指综述系统梳理规模；配套 Awesome 清单截至维护日 **564 entries**（含 VLA/数据/基准分册），宜 **综述读框架、清单查条目**。
- **WAM** 核心问题：视频里学到的运动与交互，怎样变成 **可执行、可闭环** 的机器人控制；强调 **预测参与动作学习**，而非只做反应式 VLA 或外挂世界模型。

### 与 VLA 泛化的接点

| 路线 | 代表（文内） | 扩展什么 | 仍缺什么 |
|------|--------------|----------|----------|
| 语义 + 子任务链 | π0.5 | VLM 语义、离散子任务与连续动作专家 | 子任务/动作标注与 scale |
| 人类视频动作提取 | EgoScale | 第一视角规模与相对手腕运动 | 估计误差、本体对齐 |

### 训练两阶段（文内叙事）

1. **预训练**：action-free 视频 → 时空/接触变化；构建潜在或显式动作表征；前向/逆动力学与生成目标对齐。
2. **后训练**：机器人示范微调；世界模型增广轨迹/外观；神经仿真 + RL；真机反馈修正预测–策略偏差。

### 架构与目标（与 RCL 2×2 对齐）

- **结构轴**：端到端 One Model vs 双系统 Dual-system。
- **接口轴**：Joint prediction vs IDM（plan-then-act）。
- 部署可 **训练保留预测头、推理只解码动作** 以降低在线算力。

### 开源核查（步骤 2.5，2026-09-25）

| 资源 | 结论 |
|------|------|
| 综述 PDF | **已发布** — [arXiv:2609.16074](https://arxiv.org/abs/2609.16074) |
| 项目页 / GitHub | **已开源** — [站点](https://rcl-robotics.github.io/Awesome-World-Action-Models/) · [rcl-robotics/Awesome-World-Action-Models](https://github.com/rcl-robotics/Awesome-World-Action-Models)（MIT 策展与静态站） |
| 训练代码 / 权重 | **不适用** — 综述 + 清单性质 |

## 对 wiki 的映射

- **补强（无新建实体）：** [Awesome World-Action Models（RCL）](../../wiki/entities/awesome-world-action-models-rcl.md)、[World Action Models（WAM）](../../wiki/concepts/world-action-models.md)
- **论文归档：** [rcl_wam_robot_learning_survey.md](../papers/rcl_wam_robot_learning_survey.md) — 更新 arXiv 与中文导读链
- **交叉：** [VLA](../../wiki/methods/vla.md)、[Generative World Models](../../wiki/methods/generative-world-models.md)、[π0.5 政策](../../wiki/methods/pi07-policy.md)（π0.5 文内案例）、[depth-wam](../../roadmap/depth-wam.md)

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] arXiv:2609.16074 写回 RCL 综述 source / sites / repos
- [x] 预训练–后训练与 π0.5/EgoScale 叙事写回 WAM 概念页
- [x] 项目页开源状态复核（步骤 2.5）
