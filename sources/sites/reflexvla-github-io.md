# ReflexVLA Project Page

> 来源归档（ingest）

- **标题：** Reflex: Enabling Fast and Predictive Vision-Language-Action Models for Reaction-Critical Manipulation（项目主页）
- **类型：** project site
- **官方入口：** <https://reflexvla.github.io/>
- **论文：** <https://arxiv.org/abs/2608.14379>
- **基准代码：** <https://github.com/LxRoboticsLab/ReflexBench>（主页顶部首个按钮 `ReflexBench` 指向该仓库）
- **数据集：** <https://huggingface.co/datasets/cyx337/ReflexBench_dataset>（`Dataset` 按钮指向 LeRobot v3.0 数据集）
- **策略代码：** 前端按钮 **Code After acceptance**（`title="Code will be released after the paper is accepted"`，无 `href`）
- **机构：** 上海交通大学（SJTU）
- **入库日期：** 2026-08-17
- **最后更新：** 2026-11-14
- **一句话说明：** 官方项目页：提供已开源的 `ReflexBench` 基准仓库与 `ReflexBench_dataset` 数据集入口，并展示 `ReflexVLA` 1B 模型效果与“录用后开源”策略代码占位状态。
- **通读：** 已完成
- **核对范围：** 通读离线抓取的项目主页 HTML `sites/reflexvla.html`（L1-358 全文），完整核查 Hero 区链接按钮组（5 个按钮的 `href` 与占位属性）、4 张指标卡片、ReflexBench / LIBERO / 渐进消融 / AgileX Piper 4 张 HTML 结果表及 Citation 元数据。

## 开源状态（2026-11-14 DOM 核查更新）

| 项 | 官网状态与链接 | 源码可读性 |
|----|----------------|------------|
| **Paper / arXiv** | <https://arxiv.org/abs/2608.14379>（PDF 与摘要） | 全文可公开阅读 |
| **ReflexBench 按钮** | <https://github.com/LxRoboticsLab/ReflexBench> | **已开源（BSD 3-Clause）**：Isaac Lab 六任务、数据采集与同步/异步延迟评测器（见 [`sources/repos/reflexbench.md`](../repos/reflexbench.md)） |
| **Dataset 按钮** | <https://huggingface.co/datasets/cyx337/ReflexBench_dataset> | **已发布**：LeRobot v3.0 示教集（每任务约 200 ep） |
| **Code 按钮** | 占位标签 `is-placeholder code-button`，文案 **Code After acceptance**（HTML `title="Code will be released after the paper is accepted"`，无链接） | **ReflexVLA 策略本体尚未发布** |

## 页面结构与核验内容

| 区块 | 内容要点 |
|------|----------|
| **Authors** | 上海交通大学（SJTU）：Yuxuan Chen（个人页 `elycyx.github.io`）、Wanruo Zhang、Xiao Li |
| **ReflexBench** | 6 个动态任务，仿真与控制解耦，支持同步/异步推理延迟注入 |
| **ReflexVLA** | 冻结 DINOv3 潜在未来预测 + 视觉骨干中间层时序融合 + 批量编码 & CUDA Graph 回放 |
| **Result Cards** | ReflexBench 均值 `50.4%`、参数量 `1B`、优化后推理耗时 `65.0 ms`、LIBERO 均值 `97.2%` |
| **Tables** | 完整刊载 Table I（7 模型 × 6 任务，含 DynamicVLA 37.3% 与 SmolVLA 38.3%）、LIBERO（4 模型）、渐进消融表（精确到微秒级：如基线 `81.522 ms` → 优化后 `64.991 ms`）与 AgileX Piper 真机三任务表 |

## 对 wiki 的映射

- 论文归档：[`sources/papers/reflexvla_arxiv_2608_14379.md`](../papers/reflexvla_arxiv_2608_14379.md)
- 基准仓库归档：[`sources/repos/reflexbench.md`](../repos/reflexbench.md)
- 沉淀实体：[`wiki/entities/paper-reflexvla.md`](../../wiki/entities/paper-reflexvla.md)
