# Articraft 项目主页（articraft3d.github.io）

> 来源归档

- **标题：** Articraft: An Agentic System for Scalable Articulated 3D Asset Generation
- **类型：** site（项目页 / Demo / 论文与数据集展示）
- **链接：** https://articraft3d.github.io/
- **入库日期：** 2026-05-16
- **一句话说明：** 面向「从物体描述生成可仿真可关节 3D 资产」的 **agentic 系统**：编码代理在受限工作区内用 **LLM 友好 SDK** 写 `model.py` 类程序，经 **执行 harness** 编译/探测资产并返回结构化验证信号，迭代得到带关节与运动限位、PBR 的仿真就绪对象；公开 **Articraft-10K**（万余可关节资产）及与 LiteReality、VR、图像条件生成等集成演示。
- **论文（项目页引用）：** arXiv:2605.15187（bib 条目中作者含 Zhou, Matt 等；项目页亦列出 Zheng / Rupprecht / Vedaldi / Wu 等合作方信息）
- **沉淀到 wiki：** 是 → [`wiki/entities/articraft.md`](../../wiki/entities/articraft.md)

## 项目页公开主张（归纳）

### 工作方式

- **受限代理工作区**：单一可写 `model.py`、只读 SDK 文档、精选示例、**较小动作空间**；每轮由 LLM 改程序，经 harness **编译或对当前资产做探测**，根据 **验证信号与结构化反馈** 修订下一版。
- **SDK 能力边界（叙述层）**：定义部件、组合几何、指定 **关节与运动限位**、编写 **验证测试**；目标输出为 **simulation-ready** 的可关节对象（含 PBR 材质叙事）。

### 数据与展示

- **Articraft-10K**：宣称 **10,000+** 可关节 3D 资产的大规模集合；页面提供随机子集可视化。
- **应用叙事**：图像条件生成、与 **LiteReality** 管线结合的场景级可交互重建、物理仿真开门等、**VR** 中直接加载交互。

### 对照实验（页面展示名）

项目页提供与 **Articulate-Anything**、**PhysX-Anything**、**Codex**、**URDF-Anything+** 等条目的生成结果对比卡片（以项目页当前版本为准）。

## 对 wiki 的映射

- **实体页**：[`wiki/entities/articraft.md`](../../wiki/entities/articraft.md) — 将「自然语言/描述 → 可关节 3D + 仿真」与 [文字生成 CAD](../../wiki/concepts/text-to-cad.md)、[URDF-Studio](../../wiki/entities/urdf-studio.md)、[MuJoCo](../../wiki/entities/mujoco.md) 等资产与仿真链路对照。

## 备注（维护者）

- Demo 区依赖浏览器端资源加载；**定量指标、许可与代码 API** 以论文 PDF 与 [GitHub 仓库](../repos/mattzh72-articraft.md) 为准，本页仅作项目站归纳。
