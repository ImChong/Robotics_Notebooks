# Gaussian Is Enough（arXiv:2609.27070）

> 来源归档（paper）

- **标题：** The Gaussian Is Enough: Flow-Matching Priors Do Not Help When Fine-Tuning Large Behavior Models
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.27070>
- **PDF：** <https://arxiv.org/pdf/2609.27070>
- **项目页：** <https://cxu-tri.github.io/non_gaussian_FT/>
- **入库日期：** 2026-10-18
- **线索来源：** [机器人会写控制程序之后，能否自己造训练数据？](../blogs/wechat_embodied_station_26_papers_embodiedswe_2026-10-18.md)（具身智能小站，2026-10-18）
- **一句话说明：** 在多种行为模型上比较高斯与非高斯流匹配先验，发现后者在所测微调设置中未稳定获益。

## 开源状态

- **已开源**（步骤 2.5 核查，2026-10-18）：公众号未给代码链接；项目页列出 TRI-ML/openpi-priors 与 ToyotaResearchInstitute/lbm_eval，二者均 git ls-remote 可达。

## 核心摘录（公众号转述，未核验原文表格）

- 微调大型行为模型时，换噪声先验常被当作改进手段；该工作给出否定性证据，提示先检查编码器训练。
- 系统比较高斯与非高斯先验在多种行为模型上的微调表现。
- 指标：公众号称非高斯先验未稳定获益；极少数据等条件仍留有例外空间，未给具体数字。

## 对 wiki 的映射

- [paper-gaussian-is-enough-flow-matching-priors](../../wiki/entities/paper-gaussian-is-enough-flow-matching-priors.md)
- [编程智能体造数据与 VLA 闭环评测：具身智能小站 10-18 期 26 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-18-26-papers-technology-map.md)
