# JoyAI-RA 0.5 项目页（joyai-ra-05.github.io）

> 来源归档

- **标题：** JoyAI-RA 0.5 — Scaling Robot Manipulation Learning via Dual Action Alignment
- **类型：** site / project-page
- **URL：** <https://joyai-ra-05.github.io/>
- **论文：** <https://arxiv.org/abs/2608.05674> — 归档见 [`sources/papers/joyai_ra_05_arxiv_2608_05674.md`](../papers/joyai_ra_05_arxiv_2608_05674.md)
- **机构：** Joy Future Academy / JoyAI-RA Team（京东 JD）
- **入库日期：** 2026-08-07（2026-11-14 核对项目页完整 HTML）
- **核对范围：** 2026-11-14 通读 `https://joyai-ra-05.github.io/` 完整 HTML 源码（579 行），核查外链按钮（arXiv PDF 与 `https://robotdata.jdcloud.com/` Datasets 入口）、注释隐藏区块及图表口径。
- **通读：** 已完成
- **一句话说明：** 技术报告站：异构人/仿/机数据（72K+ h）、双动作对齐 VLWA、内–外环 RL、AgiBot 真机评测与人视频缩放曲线。

## 公开信息要点（2026-11-14 核实 HTML 源码）

- **代码与权重：** **未列** GitHub / Hugging Face / 模型下载链接；仅链 arXiv。
- **数据入口与 Demos 差异：** 页头 Datasets 按钮外链至 `https://robotdata.jdcloud.com/`；HTML 中真机 Demos 视频展示区块当前被注释隐藏（标注 temporarily hidden）。
- **数据宣称：** **72K+ h** = 人视频 **53K+ h**（含自采 **EgoLive 20K+ h**）、仿真 **11K+ h**、真机 **8K+ h**；含五阶段 ego 闭环策展与 HaMeR+MANO / ORB-SLAM3 / V-JEPA 2 SemDeDup 说明。
- **架构示意：** VLM + LAC-WM + Flow-Matching Action Expert（late fusion）；canonical **130-D** 动作空间。
- **评测：** Real-World AgiBot Benchmark；seen / unseen；内–外环 RL 与 EgoLive / LAC-WM 缩放图。

## 关联资料

- 论文摘录：[`sources/papers/joyai_ra_05_arxiv_2608_05674.md`](../papers/joyai_ra_05_arxiv_2608_05674.md)
- Wiki 实体：[`wiki/entities/paper-joyai-ra-05.md`](../../wiki/entities/paper-joyai-ra-05.md)
