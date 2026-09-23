# GenRobot RealOmni-Open DataSet（项目页）

> 来源归档（项目页 + 开源核查，入库日 2026-09-23）

- **标题：** RealOmni-Open DataSet / Gen-HumanEgo
- **类型：** site / open-dataset-portal
- **链接：** <https://www.genrobot.ai/data/open-dataset>
- **机构：** 简智机器人（GenRobot）
- **入库日期：** 2026-09-23
- **一句话说明：** GenRobot 开放数据门户：宣传 **10Kh RealOmni-Open**（**1M+ clips**、真实场景 × 多样技能），当前可下载入口指向 Hugging Face 上的 **Gen-HumanEgo** 人类 egocentric 子集。

## 项目页要点（入库日快照）

- **产品族：** DAS Ego / DAS Gripper / DAS Controller — 「Sync in sense. Pro in performance.」
- **开放数据叙事：** **10Kh** 规模、**Real & Omni Scenes**、**Diverse skills**，面向 Embodied AI。
- **下载入口：** Hugging Face（页面主 CTA 指向 HF 数据集）。

## 开源状态核查（步骤 2.5）

| 类别 | 结论 | 依据 |
|------|------|------|
| **Gen-HumanEgo 数据** | **已开源** | HF `genrobot2025/Gen-HumanEgo` **ungated**；许可 **CC BY-SA 4.0** |
| **读取 / 转换工具** | **已开源** | [genrobot-ai/das-datakit](https://github.com/genrobot-ai/das-datakit) |
| **可视化** | **在线工具** | [MCAP Visualization Tool](https://monitor.genrobot.click/#/index) |
| **数据 schema 文档** | **已发布** | [DAS-Ego Data Introduction](https://docs.genrobot.ai/guides/das-ego-data-introduction) |
| **训练 / VLA 代码** | **未见** | 项目页与 HF 卡片未列官方训练栈 |
| **全量 RealOmni（含机器人侧）** | **部分 / 分批发** | 门户叙事 10Kh+1M clips；入库日可直链下载的为 **Gen-HumanEgo** 人类 ego 子集 |

## 联系方式（官网）

- Email: opendata@genrobot.ai
- X: [@GenrobotAI](https://x.com/GenrobotAI)
- Discord: <https://discord.gg/rSSb5thgu>

## 对 wiki 的映射

- [gen-human-ego-dataset](../../wiki/entities/gen-human-ego-dataset.md)
- [gen-human-ego-genrobot.md](../datasets/gen-human-ego-genrobot.md)
- [das-datakit.md](../repos/das-datakit.md)
