# DiffusionDrive: Truncated Diffusion Model for End-to-End Autonomous Driving

- **类型：** 原始论文摘录（不转存论文）
- **论文：** https://arxiv.org/html/2411.15139
- **arXiv ID：** 2411.15139
- **机构：** 华中科技大学、地平线
- **代码：** https://github.com/hustvl/DiffusionDrive
- **核对日期：** 2026-11-18
- **核对范围：** 论文§3截断扩散与§4实验、附录；官方README；原始页已有项目流程图与时序图
- **通读：** 已完成

## 方法

先提出多峰锚点轨迹，在锚点周围加噪后使用短程截断扩散去噪生成多条规划轨迹。

## 原文实验及边界

论文报告NAVSIM 88.1 PDMS，约2步去噪及RTX4090约45FPS的测试条件。

锚点质量会限制覆盖度；NAVSIM并非真实交互驾驶的全面安全验证。

## 对 wiki 的映射

- [唯一详情节点](../../wiki/entities/paper-diffusiondrive.md)
- [2026 自动驾驶十项目](../../wiki/overview/autonomous-driving-open-source-top10-2026.md)
