# 机器人该从什么数据里学习？北大等用五层「数据金字塔」给出一份配方

> 来源归档（blog / 微信公众号）

- **标题：** 机器人该从什么数据里学习？北大等用五层「数据金字塔」给出一份配方
- **类型：** blog（论文科普 / 数据配方）
- **作者：** 具身智能前沿（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s?__biz=Mzg5OTY3ODkzNg==&mid=2247494197&idx=1&sn=64976e5b44950091b38a4db1629ecb12
- **发表日期：** 2026-09-05
- **入库日期：** 2026-09-19
- **抓取方式：** wechat-article-for-ai（Camoufox）
- **原始抓取落盘：** [`sources/raw/wechat_jushen_qianyan_data_pyramid_recipe_2026-09-05.md`](../raw/wechat_jushen_qianyan_data_pyramid_recipe_2026-09-05.md)
- **一句话说明：** 科普 arXiv:2607.24744 Data Pyramid 综述：五层数据（真机/UMI/Ego-Exo/仿真/通用 VL）+ 可扩展性×对齐两轴 + 动作空间/坐标系对齐 + 三类基础模型配方。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 论文实体（canonical） | [paper-data-pyramid-embodied-manipulation](../../wiki/entities/paper-data-pyramid-embodied-manipulation.md) |
| 系列专辑 | [embodied-data-collection-to-flywheel-album](../../wiki/overview/embodied-data-collection-to-flywheel-album.md) |
| 采集术语（#1） | [embodied-data-collection-four-layers-taxonomy](../../wiki/concepts/embodied-data-collection-four-layers-taxonomy.md) |
| 纵深路线 | [depth-embodied-data](../../roadmap/depth-embodied-data.md) |

## 核心摘录（MVP）

- **两轴六维：** 可扩展性 × 机器人对齐；辅以质量、多样性、可复用性、物理保真度。
- **五层：** 真机（塔尖）→ UMI → Ego/Exo → 仿真 → 通用 VL（塔基）；综合排序，非单项单调。
- **规模陷阱：** 条数/小时/QA 对不可横比；多样性常比条数更重要。
- **配方趋势：** 真机+仿真+通用+Ego+UMI 异构混合；非「来源越多越好」。
- **对齐两关：** 动作空间语义槽 vs 零填充；几何坐标系与 TCP/增量/单位元数据。
