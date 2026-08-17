# Awesome Robot Descriptions

> 来源归档

- **标题：** Awesome Robot Descriptions
- **类型：** repo / curated-list
- **链接：** https://github.com/robot-descriptions/awesome-robot-descriptions
- **作者 / 维护：** GitHub org [`robot-descriptions`](https://github.com/robot-descriptions)（与 [robot_descriptions.py](robot-descriptions-py.md) 同组织）
- **许可：** 列表本身 **CC0-1.0**；表内每条描述文件沿用上游许可证
- **Stars：** 1,635（2026-08-17 核查）；forks 146
- **默认分支：** `main`（最近推送 2026-08-04）
- **入库日期：** 2026-08-17
- **一句话说明：** 开源 URDF / Xacro / MJCF 的策展目录：按机型族列出制造商、格式链接、许可证，以及 visual / inertia / collision 是否齐全。
- **开源状态：** **已开源**（Markdown 列表；无运行时代码）。配套可运行加载器见 [robot_descriptions.py](robot-descriptions-py.md)
- **沉淀到 wiki：** [awesome-robot-descriptions](../../wiki/entities/awesome-robot-descriptions.md)
- **选型对照：** [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)

## 步骤 2.5：源码开放核查

| 入口 | 结论 |
|------|------|
| GitHub | **已开源**：CC0-1.0 列表；CONTRIBUTING 说明如何提交新条目 |
| 项目页 | 无独立站点；Gallery 节在 README（渲染依赖 GitHub 图片） |
| 代码 | **不适用作为可运行实现** — 这是发现层，不是包 |
| 姊妹仓 | [robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) **已开源、可运行**，README 写明「Most Awesome Robot Descriptions are available」 |

## 列表结构（README 分类）

- Arms / Bipeds / Dual Arms / Drones / Educational / End Effectors / Humanoids / Mobile Manipulators / Quadrupeds / Wheeled
- 每行字段：**Name | Maker | Formats（链到具体 URDF/Xacro/MJCF）| License | Visuals | Inertias | Collisions**
- 勾选是选型信号：例如 FANUC M-710iC 标 **惯量 ✖️**；部分条目许可证为 ✖️ 或 `:heavy_minus_sign:`（网格需另下）

## 相关 Awesome

README 指向：

- [Awesome Open Source Robots](https://github.com/stephane-caron/awesome-open-source-robots)
- [Awesome URDF](https://github.com/ami-iit/awesome-urdf)（ami-iit）

## 对 wiki 的映射

- [awesome-robot-descriptions](../../wiki/entities/awesome-robot-descriptions.md)
- [robot_descriptions.py](../../wiki/entities/robot-descriptions-py.md)
- [机器人描述目录选型](../../wiki/comparisons/robot-description-catalogs.md)
- [URDF](../../wiki/concepts/urdf-robot-description.md)
