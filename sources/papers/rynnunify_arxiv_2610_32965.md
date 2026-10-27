# RynnUnify: Towards Unified Embodied Brain（arXiv:2610.32965）

> 来源归档（paper）；依据 arXiv 摘要与官方项目/Hugging Face 入口核对，2026-10-27。

- **作者：** Minghao Zhu、Bohan Hou、Kun Xiang 等（共 23 位作者，完整名单见 arXiv）
- **机构：** 阿里巴巴达摩院等，详见论文作者单位
- **提交日期：** 2026-10-23
- **论文：** <https://arxiv.org/abs/2610.32965>
- **项目页：** <https://alibaba-damo-academy.github.io/RynnUnify.github.io>
- **Hugging Face：** <https://huggingface.co/collections/Alibaba-DAMO-Academy/RynnUnify>
- **一句话说明：** 16B 统一具身模型把具身理解、长程视觉规划、agentic 推理和 WAM 机器人控制纳入同一模型家族。

## 摘要式摘录

RynnUnify 结合具身理解与交错视频—语言推理，覆盖空间推理、物理预测和基于视觉未来的长程规划。其 agentic 扩展通过强化学习选择性使用视频预测，以推理未观测状态和交互结果；world-action 扩展则把理解与预测能力用于机器人控制。论文同时介绍 Rynn-VP 与 Rynn-GEU 两项视觉规划和前瞻推理评测。

论文报告：Rynn-VP 总分较 Cosmos 3 Nano 高 9.75 分；agentic 扩展在 Rynn-GEU 上较直接推理高 6.0 分。WAM 在选取的 20 个 RoboCasa365 组合任务上成功率为 51.5%，三项真实机器人任务平均成功率为 46.7%。

## 官方资源与开放状态

项目提供官方项目页和 Hugging Face collection。用户提供的发布信息称其采用 Apache-2.0；部署和复现前应以 collection 中各模型/数据条目的实际 license 与 model card 为准。该项目页在本次核查工具中无法读取，故不据此推断其中未能验证的代码、权重或训练脚本细节。

## 对 wiki 的映射

- 具身基础模型（Embodied Foundation Models）
- 世界—动作模型（WAM）
- 视觉规划与具身推理
- 具身基础模型与世界—动作模型实体（待后续从原始来源提炼）
