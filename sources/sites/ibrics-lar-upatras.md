# 项目页：IBRICS（LAR / University of Patras）

> sources/sites 归档

- **URL：** <https://lar.upatras.gr/projects/ibrics.html>
- **类型：** 机构实验室项目页
- **全称：** Imitating Human Behaviors by Robotic Systems through Computer Vision and Machine Learning
- **机构：** 帕特雷大学自动化与机器人实验室（Laboratory of Automation and Robotics, LAR）；资助 Archimedes RC / Athena RC（Greece 2.0 MIS 5154714，NextGenerationEU）
- **核查日期：** 2026-08-18
- **一句话说明：** 把计算机视觉、机器学习与基于优化的控制接到同一条线上：从人体姿态/三维运动到机器人模仿；Humanoids 2025 已落地两条 TO 主产出（AHMP 接触发现 + SE(3) 切空间浮动基参数化对比）。

## 开源状态（步骤 2.5）

项目页本身**不挂 Code 按钮**，但两篇 Humanoids 2025 论文与下游仓库交叉可核对：

| 产出 | 开源结论 | 代码 URL |
|------|----------|----------|
| AHMP（接触序列发现 + SE(3) 切空间 TO） | **已开源**（BSD-2-Clause） | [hucebot/ahmp](https://github.com/hucebot/ahmp) |
| 浮动基参数化对比 / SE(3) 切空间 TO | **已开源**（BSD-2-Clause） | 论文声明 [upatras-lar/se3_trajopt](https://github.com/upatras-lar/se3_trajopt)；社区扩展 [yusongmin1/go2_flip_TO](https://github.com/yusongmin1/go2_flip_TO)（Go2 AMP 50 Hz 导出） |

两仓 README 均回链本项目页。AHMP 另依赖 HSL MA97（学术许可）或可改 IPOPT 线性求解器；`go2_flip_TO` 默认 conda-forge **MUMPS**，不要求 HSL。

## 项目页摘要

四条技术方向：人体姿态提取、三维人体运动分析、三维动态轨迹跟踪、操作控制器。本页 ingest 升格的是 **优化控制 / 全身 TO** 两条已发表成果，而非整条视觉模仿管线。

### 论文 1：AHMP（Tsikelis 等，Humanoids 2025）

双层优化：外层 CEM-MD 发现接触序列，内层 SE(3) 切空间全身 TO。Talos 实验：扶手走廊、烟囱 1 m / 3 m。项目页给的成功率口径（约 10 次）：扶手 100% / 中位 ~100 s；烟囱 1 m 80% / ~150 s；烟囱 3 m 50% / ~300 s。论文正文以 20 次种子、平均墙钟 <200 s（扶手 100%）为准，见 [`ahmp_humanoids_2025.md`](../papers/ahmp_humanoids_2025.md)。

### 论文 2：浮动基空间参数化对比（Tsiatsianas 等，Humanoids 2025）

同一套直接配点 TO 下对比欧拉角、三类四元数与 SE(3) 切空间。任务含 Talos 走/跳房子/大跳/倒立，G1 后空翻，Go2 侧空翻。结论：大转角任务上 SE(3) 切空间最稳；欧拉在空翻失败；四元数常收敛到「跳」而非「翻」。

## 关联

- [AHMP 论文摘录](../papers/ahmp_humanoids_2025.md)
- [SE(3) 切空间 TO 论文摘录](../papers/se3_tangent_to_arxiv_2508_11520.md)
- [ahmp 仓库](../repos/ahmp.md)
- [se3_trajopt 仓库](../repos/se3_trajopt.md)
- [go2_flip_TO 仓库](../repos/go2_flip_to.md)
- [wiki：AHMP](../../wiki/entities/paper-ahmp.md)
- [wiki：SE(3) 切空间 TO](../../wiki/entities/paper-se3-tangent-to.md)
