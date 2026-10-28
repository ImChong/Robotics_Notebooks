# Genie 4：新一代世界模拟

> 来源归档（ingest）

- **标题：** Genie 4: A new generation of world simulation
- **类型：** blog
- **作者：** Raia Hadsell、Diego Rivas、Hang Qi
- **机构：** Google DeepMind
- **日期：** 2026-10-27；**入库：** 2026-10-28
- **原文：** <https://deepmind.google/blog/genie-4-a-new-generation-of-world-simulation/>
- **官方模型页：** <https://deepmind.google/models/genie/>

## 核心内容

- Genie 4 实时生成可交互世界及音频；用户以键盘导航，并用自然语言改变天气、人物、物体或镜头。Gemini Omni 提供现实世界外观与行为基础。
- Gemini 3.8 Flash 作为循环中的引导 agent，观察画面、分解目标、检查步骤并根据场景变化调整计划。文章还展示 Gemini 4 Argon 将任务步骤转成文本提示，驱动 Genie 4 渲染 Apptronik Apollo 下跳棋的模拟。
- 机器人训练环境是潜在应用；文中展示是模拟而非机器人策略训练、真机迁移或任务成功率实证。
- 作者指出长时视觉细节和物体持久性会漂移，复杂物理、空间精度、声音准确性及长期一致性仍在研究。模型页称连续交互持续数分钟；演示片段经过剪辑。
- 研究预览先面向可信测试者，后续拟向 Project Genie 的 Google AI Ultra 用户及 Gemini API / AI Studio 早期合作伙伴开放；生成视频和音频包含 SynthID。

## 开放状态（2026-10-28）

已查看博客和官方模型页，未发现训练/推理代码、权重或数据集下载链接；不据此断言未来是否开源。文章未给出可复现的机器人策略评测或量化物理精度。

## 对 wiki 的映射

- [Genie 4 实体](../../wiki/entities/deepmind-genie-4.md)
- [世界模型功能分类](../../wiki/concepts/functional-taxonomy-world-models.md)
