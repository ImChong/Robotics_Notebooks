# AdaRoboVLG（arXiv:2609.04096）

> 来源归档（ingest）

- **标题：** Adaptive Vision-Language Grasping via Composable Foundation Priors and Generalizable Grasp Synthesis
- **简称：** AdaRoboVLG
- **类型：** paper / grasp / vision-language-grasp / cross-hand
- **arXiv：** <https://arxiv.org/abs/2609.04096>
- **PDF：** <https://arxiv.org/pdf/2609.04096>
- **项目页：** <https://adarobovlg.github.io/> — 归档见 [`sources/sites/adarobovlg.md`](../sites/adarobovlg.md)
- **代码：** 截至 2026-09-04 项目页 **未列** GitHub
- **机构：** 华中科技大学（HUST）、擎朗智能（KEENON）、北京通用人工智能研究院（BIGAI）、北京大学、字节跳动
- **入库日期：** 2026-09-04
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_9_papers_open_source_2026-09-04.md)
- **一句话说明：** 先学可跨手型的物理抓取基策略（运动学映射 + force-closure），再把空间/认知/时间理解交给可组合基础模型先验，避免与端到端抓取策略紧耦合。

## 开源状态（步骤 2.5，2026-09-04）

| 组件 | 状态 |
|------|------|
| 项目页 | 已上线（方法、仿真/真机叙事、视频） |
| GitHub / 权重 | **未见** 公开链接 |

**结论：待发布** — 可引用项目页与 arXiv。

## 核心摘录

### 摘录 1：解耦

- 基策略生成并评估物理可行抓取候选。
- 任务相关理解由专门基础模型模块以 composable priors 注入，无需重训底层抓取策略。

**对 wiki 的映射：** [paper-adarobovlg](../../wiki/entities/paper-adarobovlg.md)

## 当前提炼状态

- [x] 项目页核查（2026-09-04）
- [x] wiki 映射：`wiki/entities/paper-adarobovlg.md`
