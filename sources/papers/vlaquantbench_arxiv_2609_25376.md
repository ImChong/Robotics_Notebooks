# VLAQuantBench（arXiv:2609.25376）

> 来源归档（paper）

- **标题：** VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.25376>
- **PDF：** <https://arxiv.org/pdf/2609.25376>
- **代码：** <https://github.com/jiuyixu25/VLAQuantBench>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 统一量化格式与作用范围，在闭环任务上比较 VLA 的训练后量化（独立层组与联合量化），把成功率、校准条件与真实算子延迟一并检查。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：仓库可达（git ls-remote）；公众号称显存节省不等于推理加速，复现须自备 LIBERO 与模型权重。

## 核心摘录（公众号转述，未核验原文表格）

- 压低数值误差并不保证机器人完成任务；部署团队需要闭环成功率而不是逐层误差来决定量化方案。
- 在统一 RTN 配方下统一量化格式与作用范围，比较独立层组与联合量化，并检查校准与真实算子延迟。
- 指标：公众号转述：OpenVLA-OFT 的 LIBERO-Goal 成功率由 BF16 的 96% 降至 W8A8 的 13%（每配置 200 轮）；强调格式、层范围与校准耦合，不能推广为所有 8 位量化都失效。

## 对 wiki 的映射

- [paper-vlaquantbench](../../wiki/entities/paper-vlaquantbench.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
