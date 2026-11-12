# RoboGene：多样性驱动的机器人任务自动生成框架

- **标题：** RoboGene: Diversity-Driven Agentic Task Generation Framework for VLA Pre-training
- **论文：** https://arxiv.org/abs/2602.16444
- **HTML：** https://arxiv.org/html/2602.16444
- **版本：** 论文 v3（标题按当前 arXiv）
- **项目：** https://robogene-boost-vla.github.io/
- **数据：** https://huggingface.co/datasets/X-Humanoid/RoboGene
- **代码：** 截至 2026-11-12，未核实到公开完整的 Agent 生成代码入口；仅有论文和开放数据，不把数据下载当算法开源。
- **核对范围：** 通读 arXiv HTML 的 §1–5、方法与评测表 1–3，以及附录 A–J 的任务库配置、数据分割、对照协议、消融、OOD 和任务案例；对照项目页及数据入口，未运行代码。
- **通读：** 已完成

## 方法

- 从环境、物体、技能、本体预定义库构成的任务空间，采用 Least Frequently Used（LFU）次数统计覆盖长尾；场景与物体通过文本相似度阈值校验匹配。
- LLM Generator 生成结构化 JSON 任务；具身可执行性、创新性和约束遵从三种自反思评估器提出修订，Refiner 修复不合理步骤。
- 人类在任务实际执行时记录失败原因，LLM 摘要成语义约束并写入长期记忆，下次以 RAG 检索供修订器参考。
- 生成的是**任务描述/采集目标**，并非直接合成真实动力学动作轨迹；任务验证后，作者通过遥操作采集数据。

## 原文数字与限制

- 四种任务生成方法（Human/GPT-4o/Gemini/RoboGene）各选 300 个单臂/双臂任务、每任务收 15 条真实示教；全部对照合计 **4×300×15 = 18,000** 条，不能写成 RoboGene 单方法自动生成 18,000 条。
- 项目任务库覆盖 8 类场景、118 种技能、1137 个物体；多样性评测覆盖 **719 种独立物体**（§4.2）。
- π₀ 经各方法数据预训练后，在 5 个未见双臂任务 × 5 种 OOD 变化上，RoboGene 的平均成功率 **40.2%**，对照 Human **19.0%**、GPT-4o **15.8%**、Gemini 2.5 Pro **17.8%**（表 2，每方法 1500 次评测）。
- 使用 50 条下游示教微调，5 任务平均成功率 RoboGene **48%** vs Human **38%**（附录表 9）。
- 上限仍受人工维护的任务库和物体供应限制；自反思可降低幻觉但不能取代真实可执行验证。

## 对 wiki 的映射

- [RoboGene](../../wiki/entities/paper-robogene.md)
- [RoboAug：数据增广](../../wiki/entities/paper-roboaug.md)
