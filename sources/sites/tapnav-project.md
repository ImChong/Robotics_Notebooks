# TAPNAV 项目页（tapnav.github.io）

> 来源归档

- **标题：** TAPNAV: Humanoid Navigation through Tactile Active Perception
- **类型：** site（学术项目页）
- **URL：** <https://tapnav.github.io/TAPNAV-Project/>
- **论文：** <https://arxiv.org/abs/2610.10748>（归档：[tapnav_arxiv_2610_10748.md](../papers/tapnav_arxiv_2610_10748.md)）
- **代码：** 研究代码仍未发布——页面 “Code” 项为不可点击的 “Code coming soon”（2026-11-08 核查，基于项目页的公开 HTML 源码）
- **项目网站源仓库：** <https://github.com/TapNav/TAPNAV-Project>（仅 HTML/CSS/JS、图片及演示视频，**不是算法代码**）
- **核验入口：** <https://github.com/TapNav/TAPNAV-Project/blob/main/index.html>（`nav.paper-links` 中 Code 使用 `span.pill.is-disabled`，无下载链接）
- **入库日期：** 2026-10-12

## 页面结构

| 区块 | 内容要点 |
|------|----------|
| Abstract / Method | 位姿信念（里程计 + IMU + 触觉）、全局规划（不确定性受限路径）、局部规划（信息增益探触）、全身控制器 |
| Simulation Results | MuJoCo 四图；平均成功率 TapNav 95%、Random-touch 77.5%、Sweep-touch 38.8%、Odometry-only 22.5% |
| Real-world Results | Unitree G1，办公室走廊与厨房 |
| BibTeX | 论文引用 |

## 开源结论

**待发布（研究代码）**：项目页只有 “Code coming soon”，未列可运行的训练、推理、部署算法代码、权重或数据入口。已核验到官方项目展示站点的公开仓库，但它不构成论文复现代码，故不建算法 `sources/repos/` 条目。

## 对 wiki 的映射

- wiki/entities/paper-tapnav-humanoid-tactile-navigation.md
