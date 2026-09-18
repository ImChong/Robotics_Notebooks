# HumanCLAW 项目页（human-claw.github.io）

> 来源归档（ingest · 项目页核查）

- **URL：** <https://human-claw.github.io/>
- **标题：** HumanCLAW: Can Vision-Language Models Act Through a Body?
- **类型：** project site / benchmark
- **机构：** Meta；南洋理工大学（NTU）；华盛顿大学（UW）；布朗大学（Brown）；西北大学（Northwestern）
- **论文：** <https://arxiv.org/abs/2607.27180>
- **代码：** <https://github.com/Human-CLAW/HumanCLAW>
- **Motion 权重：** <https://huggingface.co/HumanCLAW/HumanCLAW>
- **HSSD 补充数据（gated）：** <https://huggingface.co/datasets/HumanCLAW/HumanCLAW-HSSD>
- **Leaderboard：** <https://human-claw.github.io/#leaderboard>
- **入库日期：** 2026-09-18
- **一句话说明：** 将冻结 VLM 的 **action intelligence**（每 0.5 s 原子全身技能决策）与 DiT 全身运动生成、Half-Physics 仿真执行解耦；HumanCLAW-Bench 在 41 个 HSSD 室内场景 1,218 条 find–navigate–interact 长程 egocentric 回合上横评九款 SOTA VLM。

## 开源状态（2026-09-18 项目页 + GitHub README 核查）

| 产物 | 状态 |
|------|------|
| 评测 harness / benchmark 代码 | **已开源** · Apache 2.0 · 2026-08-17 完整发布 |
| Motion 生成权重（`paper_fullval_v1`） | **已发布** · Hugging Face |
| HSSD 烘焙 mesh 补充包（1,693 实例） | **部分发布** · HF **gated** 数据集 |
| 官方 HSSD-Hab val 场景 | **需授权** · 非本仓库附带 |
| VLM 推理端点 | **自备** · OpenAI-compatible 或 queue worker |
| 论文 leaderboard 数值 | **在线更新** · 部分模型标注为 arXiv 后续评测 |

## 页面要点

- **Action intelligence：** 在闭环执行中，每步决定「身体下一步做什么」，而非低层 motor control。
- **HumanCLAW 三解耦：** VLM skill harness（感知→中层目标→原子技能）→ verifier → DiT motion continuation + per-skill ControlNet → Half-Physics（kinematic 人体 + 真实碰撞/重力/物体响应）。
- **HumanCLAW-Bench：** 任务「find `<category>` → navigate to zero distance → sit on it」；6 类目标（chair/bed/couch/potted plant/toilet/TV）；三阶段 progressive success（FindSR / NavSR / InteractSR）；难度按 geodesic distance、choice points、obstacle density 分 easy/medium/hard。
- **关键结论（项目页）：** 九 VLM 无一 solve benchmark；最佳 InteractSR **16.8%**（Gemini-3.1）；瓶颈在 egocentric self-localization、body misawareness、reach≠interact。

## 对 wiki 的映射

- [HumanCLAW 实体页](../../wiki/entities/paper-humanclaw.md)
- [代码归档](../repos/humanclaw.md)
- [论文摘录](../papers/humanclaw_arxiv_2607_27180.md)
