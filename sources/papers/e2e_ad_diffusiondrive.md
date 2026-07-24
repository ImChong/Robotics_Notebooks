# DiffusionDrive: DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving（arXiv:2411.15139）

> 来源归档（ingest）

- **标题：** DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving
- **缩写：** **DiffusionDrive**
- **类型：** paper / end-to-end autonomous driving
- **arXiv：** <https://arxiv.org/abs/2411.15139>
- **Venue：** CVPR 2025 Highlight
- **机构：** 华中科技大学 hustvl（HUST）等
- **项目页：** 无
- **代码：** https://github.com/hustvl/DiffusionDrive
- **入库日期：** 2026-07-24
- **策展来源：** [深蓝AI 端到端十大盘点](../../sources/blogs/wechat_shenlan_ai_ad_e2e_top10.md)
- **一句话说明：** 先预测多模态锚点轨迹再截断扩散去噪，把去噪步数压到约 2 步，在 NAVSIM 上冲高 PDMS 并达约 45 FPS。

## 开源状态（2026-07-24）

- **已开源** — 代码：https://github.com/hustvl/DiffusionDrive；项目页：无

## 核心摘录

- **演进线索：** 截断扩散实时规划
- **机制要点：** 见 wiki 实体页「核心原理」
- **指标索引：** NAVSIM：**88.1 PDMS**；单卡 RTX 4090 约 **45 FPS**；去噪步数相对传统约 **10×** 减少（盘点/论文）。

## 对 wiki 的映射

- 主实体：[paper-diffusiondrive](../../wiki/entities/paper-diffusiondrive.md)
- 父节点：[e2e-autonomous-driving-top10-algorithms](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)
- 策展博客：[wechat_shenlan_ai_ad_e2e_top10](../../sources/blogs/wechat_shenlan_ai_ad_e2e_top10.md)
