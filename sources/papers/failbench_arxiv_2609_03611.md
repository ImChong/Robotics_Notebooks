# FailBench（arXiv:2609.03611）

> 来源归档（ingest）

- **标题：** FailBench: How Reliable are VLMs at Judging Robot Task Success?
- **简称：** FailBench
- **类型：** paper / benchmark / failure-detection / vlm
- **arXiv：** <https://arxiv.org/abs/2609.03611>
- **PDF：** <https://arxiv.org/pdf/2609.03611>
- **项目页：** <https://metric-ai-lab.github.io/failbench/> — 归档见 [`sources/sites/failbench.md`](../sites/failbench.md)
- **代码：** <https://github.com/Metric-AI-Lab/failbench> — 归档见 [`sources/repos/metric-ai-failbench.md`](../repos/metric-ai-failbench.md)
- **机构：** 度量人工智能实验室（Metric AI Lab）
- **入库日期：** 2026-09-04
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 跨 14 个公共来源的 2197 次操作尝试，统一协议评 13 个 VLM 失败检测器；最好平均 balanced accuracy 仅 0.77。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（数字、发现、排行榜区块） |
| GitHub | 存在，Apache-2.0；内容为 `LICENSE` + `README` + `index.html`（站点镜像） |
| Code / Dataset 按钮 | 仍指向 `#`，**未见** 可运行 harness / 数据打包 |

**结论：部分开源** — 论文与项目页叙事已公开；复现入口待补。勿把空按钮写成「已发布完整评测栈」。

## 核心摘录

### 摘录 1：证据类型决定上限

- 75% 失败为自然发生；6 个真实来源本不是失败检测数据集。
- 可观察物体运动时接近饱和；接触密集装配没有任何模型超过 0.60 balanced accuracy。
- 专门微调的机器人失败检测模型整体不如通用 VLM。

**对 wiki 的映射：** [paper-failbench](../../wiki/entities/paper-failbench.md)

## 当前提炼状态

- [x] 项目页与仓库核查（2026-09-04）
- [x] wiki 映射：`wiki/entities/paper-failbench.md`
