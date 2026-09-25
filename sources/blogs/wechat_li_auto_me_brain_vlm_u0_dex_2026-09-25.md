# 理想一口气连发四项具身基座模型工作：车企进具身，理想第一次交卷

> 来源归档（blog / 微信公众号）

- **标题：** 理想一口气连发四项具身基座模型工作：车企进具身，理想第一次交卷。
- **类型：** blog
- **作者：** 具身智能研究室（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/UVSRMDa8Aq2oJtqkRUU_EA
- **发表日期：** 2026-09-25
- **入库日期：** 2026-09-25
- **抓取方式：** Camoufox + wechat-article-for-ai（`sources/raw/wechat_li_auto_me_four_papers_2026-09-25/article.md`）
- **一句话说明：** 理想汽车基础模型团队（MachEmbodied）连发 ME-Brain 1.0 / ME-VLM / ME-U0 / ME-Dex 1.0 四篇 arXiv；**4/4 均有独立 `paper-*` 详情节点**（本 ingest **新建 2**；ME-U0、ME-Dex **复用**；**0 重复 arXiv 节点**）。

## 核心摘录（归纳，非全文）

文内以 Piper 双臂真机对比（叠碗 10/10 vs 插充电器 1/10）引出四篇分工：**经验记忆与演进**、**认知与规划 VLM**、**理解–生成统一 WAM**、**异构触觉 WAM**。

### 4 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | ME-Brain-1.0 | [2609.24271](https://arxiv.org/abs/2609.24271) | **部分开源** — Focus-VLWA 动作模型训练/推理已释；完整 ME-Brain 框架与真机集成 **待发布** | [paper-me-brain-1-0](../../wiki/entities/paper-me-brain-1-0.md) |
| 02 | ME-VLM | [2609.24526](https://arxiv.org/abs/2609.24526) | **待发布** — 技术报告 + 项目页；推理/训练/权重 **TODO** | [paper-me-vlm](../../wiki/entities/paper-me-vlm.md) |
| 03 | MachEmbodied-U0 (ME-U0) | [2609.25627](https://arxiv.org/abs/2609.25627) | **已开源** `MachEmbodied/ME-U0`（**复用**） | [paper-me-u0](../../wiki/entities/paper-me-u0.md) |
| 04 | ME-Dex 1.0 | [2609.21449](https://arxiv.org/abs/2609.21449) | **部分开源** 推理 + HF 权重（**复用**） | [paper-me-dex-1-0](../../wiki/entities/paper-me-dex-1-0.md) |

## 对 wiki 的映射

- **4/4 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 2** 个实体；**2 复用** ME-U0 / ME-Dex；**0 重复 arXiv 节点**。
- 阅读坐标：[理想 MachEmbodied 四篇技术地图](../../wiki/overview/li-auto-machembodied-4-papers-technology-map.md)。
- 交叉：[World Action Models](../../wiki/concepts/world-action-models.md)、[VLA](../../wiki/methods/vla.md)、[Manipulation](../../wiki/tasks/manipulation.md)。

## 当前提炼状态

- [x] 公众号正文抓取
- [x] MachEmbodied 项目页与 GitHub 步骤 2.5 核查
- [x] 4 篇独立节点规划（2 新建 / 2 复用）
