# Imperfection for Precision: Upcycling Imperfect Data for High-Precision Robotic Manipulation

- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.26672>
- **项目页：** <https://varepsilon4p.github.io/>
- **入库日期：** 2026-09-23
- **索引来源：** [具身智能小站 12 篇盘点](../blogs/wechat_embodied_station_12_papers_collab_wm_2026-09-23.md)（<https://mp.weixin.qq.com/s/ZYnIkrJ-9H5KQw2z-AT0qQ>）
- **一句话说明：** 按 flow-matching 噪声阶段分工：低精度目标任务数据在高噪声保留语境，高精度异任务数据在低噪声传递动作精度。

## 核心摘录

1. 按 flow-matching 噪声阶段分工：低精度目标任务数据在高噪声保留语境，高精度异任务数据在低噪声传递动作精度。
2. 亚毫米级精密操作常被昂贵任务专属遥操作数据卡住；ε4P 不粗暴混合 imperfect 源，而用 flow time 作 admission filter。
3. ATX 插入 80.0%、两阶段线缆 88.3%、螺栓分拣 91.7%；相对 native co-training 最高 +31.7 pp。

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-varepsilon4p.md`](../../wiki/entities/paper-varepsilon4p.md)
- 技术地图：[`wiki/overview/collab-wm-12-papers-technology-map.md`](../../wiki/overview/collab-wm-12-papers-technology-map.md)

## 一手来源复核（2026-11-08）

- **论文正式标题**：Imperfection for Precision: Upcycling Imperfect Data for High-Precision Robotic Manipulation。
- **作者**：Hao Wei、Yang Liu、Chao Tang、Shengbao Li、Jiangtao Chen、Jinxuan Zhu、Jiaheng Wang、Hong Yin、Zhaofeng Cao、Tingguang Li。
- **投稿**：2026-09-22，arXiv:2609.26672v1，9 页、5 图。
- **方法**：目标任务低精度数据仅在高噪声区间贡献训练，异任务高精度数据仅在低噪声区间贡献训练；避免无差别混合。
- **摘要明确报告**：相比基线最高改善 31.7 个百分点；用等量 imperfect data 替代任务专属高质量数据时平均降低 4.2 个百分点。
- **核验范围**：以上由 [arXiv 官方摘要](https://arxiv.org/abs/2609.26672)直接支持。具体任务的 38.3%、48.3%、80.0%、78.3%、88.3% 需逐项核对论文实验表格；此处不将其标为已复核。完整实验、附录尚未通读。
- **源码状态**：原节点记录 2026-09-23 未开源；本次未重新确认当前官方代码发布状态，故不更新该结论。
