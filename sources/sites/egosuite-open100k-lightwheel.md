# EgoSuite-Open100K（Lightwheel 项目页）

> 来源归档

- **标题：** EgoSuite-Open100K
- **类型：** site / project-landing
- **机构：** 光轮科技（Lightwheel）
- **链接：** <https://egosuite-oepn-100k-test.lightwheel.ai/>（用户提供的测试域名；HF blog 亦引用 <https://egosuite100k.lightwheel.ai>）
- **入库日期：** 2026-09-14
- **一句话说明：** 光轮智能与 Hugging Face 联合发布的 **10 万小时级** 开放 egocentric 人类活动数据集门户；首批 **1 万小时** 已在 Hub 上线，含统计、子集导航与下载入口。

## 项目页核查（步骤 2.5 · 2026-09-14）

| 核查项 | 结论 |
|--------|------|
| **数据入口** | [HF Collection: EgoSuite-Open100K](https://huggingface.co/collections/LightwheelAI/egosuite-open100k) — `EgoStandard` / `EgoPro` / `EgoDemo` |
| **代码** | [LW-Egosuite-DevKit](https://github.com/LightwheelAI/LW-Egosuite-DevKit) — MCAP 转换与可视化（**已开源**） |
| **开放程度** | **数据已开放获取**（学术研究与商业训练许可；子集经 HF access / Bucket 分发） |
| **训练代码** | 项目页为数据门户；**无** 独立训练/推理仓库 |

## 页面要点（SPA 门户 + HF blog 交叉）

- 全量规划 **100,000 h** 第一人称人类活动；**首批 10,000 h** 已发布，余量分阶段放出。
- **15,000+** 任务 · **15,000+** 采集场景 · **7** 环境大类 · **128** 场景类型 · **18** 任务类别。
- 标注：手部姿态、身体姿态（子集）、部分子集含 **事件级语义** 标注。
- 格式：**LeRobot v3**（可 Hub 流式训练）与 **MCAP**（机器人/多模态管线）。
- 采集：全球分布式采集者 + 标准化流程；与 **EgoVerse** 联盟对齐采集/标注/共享规范。

## 对 wiki 的映射

- 主实体：[EgoSuite-Open100K](../../wiki/entities/egosuite-open100k.md)
- 工具链：[LW-Egosuite-DevKit](../../wiki/entities/cn-os-lw-egosuite-devkit.md)
- 对照：[Ego4D](../../wiki/entities/paper-ego4d.md)、[EgoScale](../../wiki/methods/egoscale.md)、[EgoWorld-100W](../../wiki/entities/egoworld-100w.md)

## 交叉链接

- HF blog：[EgoSuite-Open100K 官方介绍](../blogs/hf_lightwheel_egosuite_open100k.md)
- HF collection：[egosuite-open100k](../sites/hf-egosuite-open100k-collection.md)
- 数据集卡：[EgoStandard](../datasets/lightwheel-egostandard.md) · [EgoPro](../datasets/lightwheel-egopro.md)
- 代码：[LW-Egosuite-DevKit](../repos/lw-egosuite-devkit.md)
