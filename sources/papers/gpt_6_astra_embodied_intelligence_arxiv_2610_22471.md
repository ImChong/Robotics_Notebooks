# GPT-6-Astra Embodied Intelligence（arXiv:2610.22471）

> 来源归档（paper）

- **标题：** Has GPT-6-Astra Achieved Embodied Intelligence?
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2610.22471>
- **PDF：** <https://arxiv.org/pdf/2610.22471>
- **项目页：** <https://www.sensenova.ai/has-gpt-6-astra-achieved-embodied-intelligence>
- **代码：** <https://github.com/OpenSenseNova/SenseNova-RoboRSI>
- **入库日期：** 2026-11-01
- **线索来源：** [模型还在思考，目标已经移动：机器人评测该把时间算进来吗？](../blogs/wechat_embodied_station_56_papers_libero_dynamic_latency_eval_2026-11-01.md)（具身智能小站，2026-11-01）
- **一句话说明：** 报告在上下文示范和扩展步数下测通用模型，并比较控制框架与真机表现；提示基准能力需连同耗时、示范成本及失败恢复衡量，不能直接等同实用具身智能。

## 开源状态

- **部分开放**（2026-11-03）：官方 README 列出 runtime、多项基准集成、ICL 包与 27B Preview 权重；RoboDojo、SIMPLE 等集成及 RoboRSI 独立技术报告仍待发布。

## 核心摘录（公众号转述，未核验原文表格）

- 具身基准高分，离实际部署还有多远。
- 报告在上下文示范和扩展步数下测通用模型，并比较控制框架与真机表现；提示基准能力需连同耗时、示范成本及失败恢复衡量，不能直接等同实用具身智能。
- 指标：公众号未给具体数字，本页不引用。

## 对 wiki 的映射

- [paper-gpt-6-astra-embodied-intelligence](../../wiki/entities/paper-gpt-6-astra-embodied-intelligence.md)
- [动态评测、技能编译与环境潜变量适配：具身智能小站 11-01 期 56 篇阅读坐标](../../wiki/overview/embodied-station-2026-11-01-56-papers-technology-map.md)

## 2026-11-03 原文核查

- [arXiv v2](https://arxiv.org/html/2610.22471v2) 与 [PDF](https://www.sensenova.ai/gpt-6/assets/docs/has_gpt-6_achieved_embodied_intelligence.pdf)；已核查正文方法、消融及 G1 真机附录，其他附录尚待逐项通读。
- 知识包由训练示范构建并在测试前冻结；扩展预算继续同一回合，且 BEHAVIOR 仅用标准预算。
- G1 抓放 2/3、行走 3/3，分别评估，成功抓放耗时超过一小时；不能视作实时控制可靠性。
