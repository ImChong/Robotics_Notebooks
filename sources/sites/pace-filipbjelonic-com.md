# PACE Documentation（pace.filipbjelonic.com）

- **类型：** 项目文档站（MkDocs + Material for MkDocs）
- **入口：** <https://pace.filipbjelonic.com/>
- **主体：** Filip Bjelonic / ETH Zurich Robotic Systems Lab（PACE 框架）
- **关联仓库：** <https://github.com/leggedrobotics/pace-sim2real>
- **关联论文：** arXiv:2509.06342
- **收录日期：** 2026-07-15

## 一句话

**PACE**（Precise Adaptation through Continuous Evolution）面向多样足式机器人的 **系统化 sim2real 管线** 官方文档：统一 **执行器建模**、**自动系统辨识** 与 **RL 控制器真机部署** 工作流。

## 为什么值得保留

- 论文方法与开源代码之间的 **操作层桥梁**（安装、示例、指南、最佳实践）。
- 明确 PACE 全称与定位，与 arXiv 正文缩写 **PACE** 一致。
- 提供 **引用 BibTeX** 与 ANYmal 等示例入口，便于 wiki 溯源。

## 文档站公开结构（2026-07-15）

### 定位（首页）

> PACE provides unified tools for accurate actuator modeling, automatic system identification for seamless deployment of RL controllers to real hardware.

### 主要分区

| 分区 | 说明 |
|------|------|
| Getting started | 前置条件、安装、首次运行 |
| Examples | 公开 ANYmal 版上的参数辨识与部署最小示例 |
| Guides / Basics | 搭建自有 PACE 环境 |
| Guides / Advanced | 自定义目标函数、参数与优化 |
| Best practices | 实践建议与常见陷阱 |
| Concepts / API Reference / Development | 规划中 |

### 引用（站点提供）

```bibtex
@article{bjelonic2025towards,
  title         = {Towards Bridging the Gap: Systematic Sim-to-Real Transfer for Diverse Legged Robots},
  author        = {Bjelonic, Filip and Tischhauser, Fabian and Hutter, Marco},
  journal       = {arXiv preprint arXiv:2509.06342},
  year          = {2025},
  eprint        = {2509.06342},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

## 沉淀到 wiki

- [wiki/entities/paper-pace-sim2real-legged-robots.md](../../wiki/entities/paper-pace-sim2real-legged-robots.md)
