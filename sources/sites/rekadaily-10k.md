# RekaDaily-10k（研究发布页）

> 来源归档

- **标题：** RekaDaily-10k: Collecting 10,000+ Hours of Egocentric Household Manipulation Data
- **类型：** site / research announcement
- **链接：** <https://reka.ai/labs/research/rekadaily-10k-egocentric-household-manipulation-data>
- **备用入口（HF README 引用）：** <https://reka.ai/news/rekadaily-10k-egocentric-household-manipulation-data>
- **Hugging Face（raw tier）：** <https://huggingface.co/datasets/RekaAI/RekaDaily-10k-raw>
- **数据引擎：** [Claru](https://claru.ai)
- **机构：** 瑞卡人工智能（Reka AI）
- **发布日期：** 2026-08-06
- **入库日期：** 2026-08-07
- **许可：** Apache 2.0（ungated；含商用与再分发）
- **一句话说明：** Reka 经 Claru 付费采集网络发布的 **10,000+ 小时** 无剧本第一人称家务视频；分 **raw**（按录制原样）与 **processed & captioned**（短片段 + 机器字幕）两档，服务世界模型 / VLA / 家用机器人视觉先验。

---

## 发布要点

| 维度 | 内容 |
|------|------|
| 目标规模 | **10,312** 小时（raw 档位全量目标；按官方说明分批上线） |
| 4K 份额 | 约 **1,670** 小时原生 4K |
| 采集方式 | 付费采集者在自有住宅用头戴/手持手机录制日常家务，**无剧本** |
| Raw 档 | 未切分、未过滤（仅基础完整性），便于自建 clipping / 标注管线 |
| Processed 档 | 经 QC → 切短片段 → 去重 → 字幕，开箱即用语言监督 |
| 生态对照 | Ego4D（日常活动基准）、Egocentric-10K（工业第一人称规模）、EPIC-KITCHENS（厨房活动金标准） |

## 采集与隐私叙事

1. **Collector consent** — 付费承包、知情录制。
2. **Bystanders** — 要求其他成人同意；无法保证时尽量出画；可识别非参与者进入人工复核。
3. **Automated PII screen** — 镜面/屏幕人脸、处方签、邮件等；命中后人工决定脱敏或丢弃。

筛选不完美；发现问题可联系发布方下架。

## 处理管线（processed 档）

```text
Submission → Quality control（信号 + 内容两遍）→ Clip & caption → Processed tier
```

- Raw 档直接来自 Submission，不经上述 QC/切分。
- QC 偏好 **假拒绝最差**：临界样本进人工复核，而非直接丢弃。
- 字幕强调 **活动在整段会话弧中的位置**（含中断与长间隔），而非单帧场景描述。

更完整的世界模型数据管线说明见：[World Model Data Pipeline](https://reka.ai/news/world-model-data-pipeline)。

## 开源状态（项目页核查，2026-08-07）

| 产物 | 状态 | 入口 |
|------|------|------|
| Raw 数据集 | **已开源**（增量发布中） | [RekaAI/RekaDaily-10k-raw](https://huggingface.co/datasets/RekaAI/RekaDaily-10k-raw) |
| Processed & captioned | **宣称另档发布**（同 `RekaDaily-10k` 前缀） | 研究页说明；入库日以 raw 为主入口 |
| 训练/推理代码 | **不适用**（本发布为数据语料，非方法仓） | — |
| 许可 | Apache 2.0，ungated | HF card |

## 对 wiki 的映射

- **wiki/entities/rekadaily-10k-dataset.md** — 数据集实体页（主升格）
- **sources/datasets/rekadaily-10k-raw.md** — HF raw 档数据卡归档
- **wiki/overview/ego-category-01-data-collection.md** — Ego 数据采集旁路对照
- **wiki/methods/egoscale.md** / **wiki/entities/egoworld-100w.md** — 大规模 egocentric 人视频生态对照
- **wiki/queries/humanoid-training-data-pipeline.md** — 人体视频层候选来源
- **wiki/entities/paper-data-pyramid-embodied-manipulation.md** — 金字塔第 ③ 层 Ego/Exo 代表补充
