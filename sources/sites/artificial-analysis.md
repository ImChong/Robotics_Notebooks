# Artificial Analysis 官方平台与数据 API

> 来源归档

- **标题：** Artificial Analysis — Independent analysis of AI models
- **类型：** site（模型评测与数据平台）
- **URL：** <https://artificialanalysis.ai/>
- **评测方法：** <https://artificialanalysis.ai/methodology>
- **数据 API：** <https://artificialanalysis.ai/data-api>
- **API 文档：** <https://artificialanalysis.ai/api-reference>
- **入库日期：** 2026-10-09
- **一句话说明：** 独立汇总并测量 AI 模型及推理服务商，在能力评测、价格、速度、首 token 延迟和可用性等维度提供比较页面与结构化数据。
- **代码：** 官网未链接可供自托管的平台代码仓库；该平台通过网站和需 API key 的数据接口提供服务。
- **沉淀到 wiki：** [`wiki/entities/artificial-analysis.md`](../../wiki/entities/artificial-analysis.md)

## 官网定位与覆盖范围

Artificial Analysis 将自身定位为 AI 模型的独立分析平台，面向模型与服务商选型。页面覆盖语言模型以及图像、视频、语音等生成模型；不同产品页提供榜单、评测、定价和性能信息。指标与模型清单持续更新，具体版本应以各页面和 API 返回为准。

语言模型信息包括模型及提供方身份、独立评测分数、输入/输出价格、输出速度和首 token 延迟等。平台官网还提供聚合指数与单项 benchmark，指数的组成及版本会变化，因此引用排名时应同时记录评测版本和观察日期，避免把动态页面当成固定事实。

## 方法与接口要点

- 官网的方法文档分别说明能力评测与服务性能/成本测量；性能数据描述通过不同服务商访问模型时的用户侧表现，不应直接解释成模型在最佳硬件上的理论上限。
- 速度指标统一 token 口径以便跨提供方比较；价格可按输入、输出或混合用量情景查看。比较时仍需核对提示长度、并发、模型版本、服务商和统计窗口。
- Free API 需创建账户并生成 key；文档说明接口受每日请求限额约束，并要求引用来源、妥善保管 key。商业 API 提供更完整数据与额外使用权。套餐、字段和限额可能变化，调用前应以当前 API 文档为准。
- API 返回模型、创建方、评测、价格与性能等结构化字段；文档建议使用稳定 ID 而非易变的名称或 slug 作为主键。

## 与机器人研究的适用边界

该平台可辅助挑选用于代码生成、研究助手、任务规划或 VLA 上游语言/视觉语言组件的基础模型，尤其适合初筛模型能力、托管推理延迟和 API 成本。它不是机器人策略或具身任务 benchmark：通用语言/多模态分数不能替代成功率、接触安全、时序稳定性、sim-to-real 泛化等机器人指标。最终候选仍需在目标机器人、任务分布、控制频率和部署硬件上验证。

## 对 wiki 的映射

- 主实体页：[Artificial Analysis](../../wiki/entities/artificial-analysis.md)
- 通用 agent 评测背景：[AI Agent 评测](../../wiki/concepts/ai-agent-evaluation.md)
- 具身基准选型入口：[具身评测基准选型闭环](../../wiki/overview/hub-embodied-eval-benchmark.md)
