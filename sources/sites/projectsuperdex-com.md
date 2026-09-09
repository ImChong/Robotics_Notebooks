# Project SuperDex 项目站

> 来源归档

- **标题：** Project SuperDex
- **类型：** site
- **机构：** Meta Platforms, Inc. / facebookresearch
- **链接：** https://projectsuperdex.com/
- **GitHub：** https://github.com/facebookresearch/project_superdex
- **入库日期：** 2026-09-09
- **一句话说明：** Meta 面向接触密集型灵巧操作的研究平台官网：四模块（Physics / Robotics / Studio / Lab）端到端管线，VR 遥操作计划 Q4 2026 发布。
- **代码：** https://github.com/facebookresearch/project_superdex（**已开源**，Apache 2.0；资产与文档 CC BY 4.0）
- **沉淀到 wiki：** 是 → [`wiki/entities/project-superdex.md`](../../wiki/entities/project-superdex.md)

---

## 站点结构（2026-09-09 核查）

| 模块 | 文档入口 |
|------|----------|
| SuperDex Physics | https://projectsuperdex.com/physics/docs/overview/ |
| SuperDex Robotics | https://projectsuperdex.com/robotics/docs/overview/ |
| SuperDex Studio | https://projectsuperdex.com/studio/ 、 https://projectsuperdex.com/studio/docs/overview/ |
| SuperDex Lab | https://projectsuperdex.com/lab/ 、 https://projectsuperdex.com/lab/docs/overview/ |

站点 meta description：**contact-rich robotics** 端到端管线——场景 authoring、仿真、遥操作、策略训练。

---

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| 项目页 → GitHub | **已开源**；README 与站点均链到 `facebookresearch/project_superdex` |
| 许可证 | 源码 Apache 2.0；资产/文档 CC BY 4.0；`superdex_mesh_cli` 为 GPLv3（OCCT/CGAL） |
| PyPI | `uv pip install superdex`（Python 3.12 预编译 wheel） |
| 分支 | `stable` 对齐最新发布 |
| 遥操作 | **SuperDex Teleop** 标 Q4 2026；Quest 3 端侧 C++，README 称 UE5 虚拟遥操作组件 |
| 论文 | README 写「Citation details will be added here upon publication」——截至入库日 **无正式论文引用** |

---

## 对 wiki 的映射

- [Project SuperDex](../../wiki/entities/project-superdex.md)
- [Contact-Rich Manipulation](../../wiki/concepts/contact-rich-manipulation.md)
- [DexBench](../../wiki/entities/dexbench.md) — 工业灵巧规格对照（不同赛道）
