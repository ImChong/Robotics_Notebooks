# Gemini Robotics（Google DeepMind 产品页）

> 来源归档（site / 产品入口）

- **标题：** Gemini Robotics
- **类型：** site / product
- **官方入口：** <https://deepmind.google/models/gemini-robotics/>
- **ER 2 子页：** <https://deepmind.google/models/gemini-robotics/embodied-reasoning/>
- **ER 2 Model Card：** <https://deepmind.google/models/model-cards/gemini-robotics-er-2/>
- **最新发布（2.5）：** <https://deepmind.google/blog/gemini-robotics-2-5-accelerates-real-world-physical-intelligence/>
- **VLA 2.5 产品页：** <https://deepmind.google/models/gemini-robotics/vla/>
- **历史发布（2）：** <https://deepmind.google/blog/gemini-robotics-2-brings-whole-body-intelligence-to-robots/>
- **开发者博文（ER 2）：** <https://blog.google/innovation-and-ai/models-and-research/google-deepmind/gemini-robotics-er-2/>
- **安全技术报告：** <https://storage.googleapis.com/deepmind-media/gemini-robotics/Gemini-Robotics-2-Safety.pdf>
- **入库日期：** 2026-07-31
- **复核日期：** 2026-11-11
- **一句话说明：** Google DeepMind 机器人智能层产品入口：当前主推 **Gemini Robotics 2.5 VLA / ER 2 / On-Device 2**，覆盖全身灵巧操作、具身推理与端侧适配；2.5 VLA 仍面向机器人合作伙伴。

## 开源状态（截至 2026-11-11）

| 组件 | 开放程度 | 说明 |
|------|----------|------|
| Gemini Robotics ER 2 | **部分开源 / API 可用** | Public preview：Gemini API、Google AI Studio；Enterprise Agent Platform private preview |
| 编排与 Spot 样例 | **已开源** | [`google-gemini/robotics-samples`](https://github.com/google-gemini/robotics-samples)（Apache-2.0）→ [repos/google-gemini-robotics-samples.md](../repos/google-gemini-robotics-samples.md) |
| Gemini Robotics 2.5（VLA） | **未开源权重** | 2026-11-10 公布，面向 robotics partners 滚动开放；产品页未提供 VLA 源码 / 权重 / 公众 API |
| Gemini Robotics 2（旧版 VLA） | **未开源权重** | 历史版本；勿混同 2.5 评测结果 |
| Gemini Robotics On-Device 2 | **未开源权重** | Trusted Tester / early-access |
| 安全报告 PDF | **已公开** | Gemini-Robotics-2-Safety.pdf |
| ASIMOV-Agentic / ASIMOV-Jailbreak | **已公开评测脚本和数据** | [Hugging Face google/asimov_agentic](https://huggingface.co/datasets/google/asimov_agentic) 提供 `asimov_physical_safety_eval.py`、`physical_safety_duo/`，不包含 GR 2.5 权重 |

- **代码：** <https://github.com/google-gemini/robotics-samples>（ER Live API / Spot / Tinybot 样例；**不是** VLA 训练仓）
- **文档：** <https://ai.google.dev/gemini-api/docs/robotics-overview>

## 对 wiki 的映射

- [gemini-robotics](../../wiki/entities/gemini-robotics.md)
- 2.5 官方发布归档：[blogs/gemini_robotics_25_general_dexterity.md](../blogs/gemini_robotics_25_general_dexterity.md)
- 2.0 旧版发布归档：[blogs/gemini_robotics_2_whole_body.md](../blogs/gemini_robotics_2_whole_body.md)
