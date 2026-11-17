# Karpo Decision 全尺寸权重与官方模型卡归档

- **标题：** Karpo Decision — Multimodal Decision Model Family（0.8B–31B）
- **类型：** model / 官方 Hugging Face 模型卡、集合
- **发布方：** MachinePulse（MachinePulse Pte. Ltd.）
- **官方集合：** https://huggingface.co/collections/MachinePulseAI/karpo-decision
- **31B 模型卡：** https://huggingface.co/MachinePulseAI/karpo-decision-31B
- **0.8B 模型卡：** https://huggingface.co/MachinePulseAI/karpo-decision-0.8B
- **12B 模型卡：** https://huggingface.co/MachinePulseAI/karpo-decision-12B
- **代码：** https://github.com/machinepulse-ai/karpo-decision
- **归档日期：** 2026-11-17；模型卡及许可为可能更新的在线版本，复用前再次核实
- **核对范围：** 阅读官方模型集合中 6 个 checkpoint 清单、31B/12B/0.8B 模型卡的模型定位、工作原理、结果、使用方式、限制和授权小节；交叉阅读推理仓 README 与实际推理代码。未下载权重或重复执行 Decision Index。
- **通读：** 已完成
- **开放状态：** 六款权重仓均有公开模型卡和权重文件；推理代码单独开源，未据此推断训练数据与端到端训练流水线开放。
- **沉淀到 wiki：** [Karpo Decision 全尺寸决策模型家族](../../wiki/entities/karpo-decision.md)

## 已核实的型号

| checkpoint | 基座 | 模型卡参数量 | 模型卡 |
| --- | --- | ---: | --- |
| 0.8B | Qwen3.5 0.8B | 0.85B | https://huggingface.co/MachinePulseAI/karpo-decision-0.8B |
| 2B | Qwen3.5 2B | 2.2B | https://huggingface.co/MachinePulseAI/karpo-decision-2B |
| 4B | Qwen3.5 4B | 4.5B | https://huggingface.co/MachinePulseAI/karpo-decision-4B |
| 9B | Qwen3.5 9B | 8.4B | https://huggingface.co/MachinePulseAI/karpo-decision-9B |
| 12B | Gemma 4 12B | 12.0B | https://huggingface.co/MachinePulseAI/karpo-decision-12B |
| 31B | Gemma 4 31B | 31.3B | https://huggingface.co/MachinePulseAI/karpo-decision-31B |

## 机制与评测摘录

- **接口：** 状态（文本、JSON、可选图像）及 typed questions：`choice`（至多 255 个候选）、`noul`（布尔）、`score`（分级）。直接给各选项概率，不续写文本。
- **Readout：** 对每个问题的读取位置以训练后的线性读出层给 255 个候选代码打分，仅对该问题合法选项 softmax。
- **31B / 12B（Gemma 4）：** 问题清单 + 状态共享块与问题独立分支；分支只读共享块及自身，不读其他分支。
- **0.8B 等 Qwen3.5：** 官方小模型卡描述为 question-list-first 的混合注意力；全注意力层双向读序列，线性注意力层按顺序读，不能不加区分地套用 Gemma 的分支注意力描述。
- **31B 声明成绩：** 2026-11-12 的 Decision Index 0.4.1 私有 Full 73.9，与 GLiDE 27B 按 0.9 分容差同列 #1；公开套件（项目自跑）public index 75.0，140,178 次有效计分。单卡 median latency 94ms 为该榜口径，非机器人执行周期。
- **局限：** 总有一个预先给定选项作为答案；单问题与状态超过 32,768 tokens 时拒绝；公开成绩不等于机器人任务成功率；小模型卡明确图像能力未在其所述评测中验证。

## 授权差异（按日期区分）

- **代码仓：** GitHub LICENSE / pyproject.toml 为 Apache-2.0。
- **权重仓：** 官方 31B 最新模型卡的许可段落记录 **2026-11-12 起**权重改为 `MachinePulse Community License`，含向第三方提供商业产品/服务时的署名要求及大营收实体的生产商用协议条件；在该日之前合法获取的 Apache-2.0 副本沿用原授权。
- **页面版本差异：** GitHub README、部分 Hugging Face 网页展示缓存仍可能出现 `Apache-2.0` 文案；不能仅据旧 README 宣称当前所有模型权重可无条件按 Apache-2.0 使用。落地前应核查各 checkpoint **当前** LICENSE、模型卡历史和具体法律条款。

## 对 wiki 的映射

- [Karpo Decision](../../wiki/entities/karpo-decision.md)：机制、实现调用、权重规格、开源/使用风险
- [AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)：结构化概率决策与独立环境结果评测的区分
- [源码归档](../repos/karpo-decision.md)：实际 API/模型执行路径及开源代码
