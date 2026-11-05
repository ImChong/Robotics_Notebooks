# GPT-6-Astra Embodied Intelligence 官方仓库

> 来源归档（repo）

- **标题：** Has GPT-6-Astra Achieved Embodied Intelligence?
- **类型：** repo
- **链接：** https://github.com/OpenSenseNova/SenseNova-RoboRSI
- **arXiv：** <https://arxiv.org/abs/2610.22471>
- **入库日期：** 2026-11-01
- **一句话说明：** 报告在上下文示范和扩展步数下测通用模型，并比较控制框架与真机表现；提示基准能力需连同耗时、示范成本及失败恢复衡量，不能直接等同实用具身智能。
- **沉淀到 wiki：** [`wiki/entities/paper-gpt-6-astra-embodied-intelligence.md`](../../wiki/entities/paper-gpt-6-astra-embodied-intelligence.md)

## 开源状态

- **部分开放**（2026-11-03）：runtime、若干基准集成和 27B Preview 权重已公开；RoboDojo、SIMPLE 等集成及独立技术报告待发布。源码快照 `407a0041f8273500fc654b5ee5683d2d1b14cd5c`；评测入口 `scripts/evaluation/evaluate_model.py` 调用 `roborsi.evaluation.cli`。
