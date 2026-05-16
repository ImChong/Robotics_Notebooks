---
type: concept
tags: [cad, generative-ai, hardware, design, api, manufacturing, llm, robotics]
status: complete
date: 2026-05-14
updated: 2026-05-16
related:
  - ../entities/urdf-studio.md
  - ../entities/atom01-hardware.md
  - ../entities/articraft.md
  - ./sim2real.md
sources:
  - ../../sources/sites/text-to-cad-tools.md
summary: "文字生成 CAD 已从纯研究演示进入可用早期：适合概念件与参数化初稿，复杂装配与生产级 DFM 仍依赖专业 CAD；机器人方向更稳的是 LLM + CadQuery/OpenSCAD 参数化脚本再导出 STEP。"
---

# 文字生成 CAD（Text-to-CAD）

**文字生成 CAD** 指用**自然语言提示**或**对话式代理**，自动生成或迭代**可编辑的 CAD 几何**（常见为 **B-rep** 实体），并通常导出 **STEP**、**STL**、**DXF** 等交换格式；其工程目标偏向**加工、公差与装配**，而不是仅输出可视化网格。

## 成熟度与典型用途（2026 前后共识）

整体判断：**已从研究演示进入「可用」阶段，但仍偏早期**。多数产品更适合：

- **快速概念建模**与**机械结构初稿**
- **参数化零件草图**（孔阵列、法兰、支架、简单壳体）
- **3D 打印原型**与实验夹具
- **导出 STEP / STL / DXF** 后在传统 CAD 里二次加工

仍**很难**在无强人工审图的前提下，**完全替代**专业 CAD 工程师完成：**复杂装配**、**全公差链**、**大规模约束与可靠性设计**、**产线级 DFM**。

## 为什么对机器人研究重要

- **硬件与夹具链路**：从概念支架、传感器支架到末端执行器零件，研发流程仍大量依赖 **STEP / 参数化模型** 与供应链沟通；若 LLM 参与设计，需要知道「生成的是否为制造级模型」。
- **与仿真资产的关系**：仿真常用的 **URDF / MJCF** 多消费 **网格与惯性近似**；STEP 往往是**上游真值**或**加工真值**，中间通常还有 **CAD → 网格 / URDF** 的转换与简化（见 [URDF-Studio](../entities/urdf-studio.md) 等工具链）。另一条相邻路线是 **程序化可关节 3D**（例如 [Articraft](../entities/articraft.md) 所归纳的 agent + SDK + harness），侧重带关节与验证闭环的 **仿真就绪网格资产**，与制造向 B-rep 流程目标不同但常在同一仿真管线中汇合。
- **Sim2Real 提示**：仿真里碰撞几何若与加工件不一致，接触与质量属性会漂移；文字生成 CAD 若用于零件原型，仍要做**独立几何校验与测量**（参见 [Sim2Real](./sim2real.md) 中关于模型一致性的讨论）。

## 能力分野（选型心智模型）

| 维度 | 制造向 Text-to-CAD / 脚本 CAD | 常见 Text-to-3D（网格） |
|------|--------------------------------|-------------------------|
| 典型输出 | STEP、带特征或脚本的参数模型 | OBJ、STL、GLB 等 |
| 可编辑性 | 强调尺寸、约束与工程修改 | 多为网格编辑 / 重拓扑 |
| 与 CNC / 公差分析 | 厂商或脚本栈常以此为主叙事 | 通常较弱或非目标 |
| 与游戏 / 渲染 | 可选 | 主战场 |

## 代表工具谱系（简评）

详细链接见 [sources/sites/text-to-cad-tools.md](../../sources/sites/text-to-cad-tools.md)。

### 1. Zoo Design Studio（KittyCAD）

业内公开讨论度较高的 **AI-native 机械 CAD** 路线之一。

- **特点**：自然语言生成 CAD；强调 **可编辑 B-rep / STEP**（而非仅 mesh）；**参数化**与 **Zookeeper** 对话式 CAD Agent；面向机械设计工作流。
- **示例 Prompt（英文，偏行星壳体类零件）**：

```text
Generate a 60mm diameter planetary gearbox housing with four M4 mounting holes.
```

- **适合**：机器人结构件、机械零件、**3D 打印**件、工业设计草模。
- **相对优势**：公开叙事强调「真 CAD 几何体」而非纯三角网格；可与 **KCL** 脚本层互操作（见官方文档）。

### 2. Adam（adamcad.com，口语里常被称作 AdamCAD）

偏 **「在现有 CAD 里干活的 AI 助手」** 而非从零替代整套 CAD。

- **特点**：公开材料强调 **Part editing**、**选区上下文**、**特征树整理**、**参数化串联变量**；提供 **Onshape 扩展**与 **Fusion 扩展**等宿主集成（以应用商店 / 安装页为准）。
- **适合**：机械图纸迭代、在成熟 CAD 会话里用自然语言替代大量细碎点击；对「工程化」预期通常高于纯玩具型 demo。

### 3. Autodesk Fusion 等「AI 增强 CAD」

Autodesk 产品线正在把 **AI** 深度嵌入 **Fusion / Maya / Flow** 等工具链：例如 **生成式设计**、**工程图辅助**、**绘图助手**、部分 **text-to-3D** 能力等。

- **定位**：更贴近 **「AI 增强的专业 CAD」**——在约束求解、图纸、生成式拓扑等环节减负；**不等价于**「一句话端到端生成可投产的全机数字样机」。

### 4. OpenSCAD + LLM（Claude / GPT / Cursor 等）

许多工程师在真实项目里采用的 **脚本化参数 CAD** 路线：

**自然语言 → LLM 生成 OpenSCAD / 类似 DSL → 本地或 CI 中生成几何 → 导出 STEP/STL**。

- **优点**：**参数化极强**；**Git / 代码审查**友好；特别适合 **机器人支架、孔位阵列、模块化件**；LLM 对「写程序」往往比对「直接猜 B-rep 拓扑」更稳定。
- **同类 Python 栈**：[CadQuery](https://cadquery.readthedocs.io/)、[Build123d](https://build123d.readthedocs.io/)（均常见基于 OCCT 的 B-rep 与 STEP 导出，以各自文档为准）。

### 5. CadQuery + AI Agent（学术与开源热点）

大量 **Text-to-CAD** 研究与工程原型共享同一抽象：

**自然语言 → LLM 生成 CadQuery（或类似）Python → 执行后得到 CAD 实体**。

- **原因**：CAD 在实现层往往就是 **程序 + 约束**；LLM 对结构化代码与 API 调用的适配，常优于「一次对话直接闭合复杂 B-rep」。
- **实践提示**：把 **「生成代码」** 与 **「执行与导出」** 拆成可测试步骤（lint、单元几何断言、CI 导出 STEP），比单次黑盒生成更接近工业习惯。

### 6. 面向「3D 资产 / 网格」而非「工业 CAD」的工具

典型取向：**外观、游戏与动画资产、视觉原型**，输出以 **mesh / 纹理** 为主，例如 **Tripo**、**腾讯混元 3D（Hunyuan3D）**、**Meshy**、**Wonder3D** 等路线（见来源索引中的链接）。

- **更适合**：概念造型、渲染、部分粗打印实验。
- **不适合默认承担**：精密机器人承力结构、**公差与配合设计**、**可制造性闭环**、**大装配约束**——这些仍应回到 B-rep / 专业 CAD 或脚本 CAD 工作流。

## 机器人方向：更推荐的工程路线

对 **腿足 / 人形 / 夹具** 等场景，更值得投入的是：

**LLM + 参数化 CAD（脚本优先）**，而不是赌 **「一句话生成整机复杂装配」**。

更现实的闭环是：

```mermaid
flowchart LR
  NL[自然语言需求] --> LLM[LLM 生成 CadQuery / OpenSCAD / Build123d]
  LLM --> REV[人类调参与几何审阅]
  REV --> EXP[导出 STEP / STL]
  EXP --> PRO[Fusion / SolidWorks 等二次设计]
```

要点：

- **人类在环**：把 LLM 放在「加速草稿与参数探索」的位置，而不是「一次性签发生产图纸」。
- **可验证**：脚本可 diff、可跑测试（体积、孔距、壁厚下限），比单次对话 mesh 更易做回归。
- **与供应链衔接**：STEP 进入企业 CAD 后，再补 **工程图、公差、表面处理与工艺审查**。

## 当前能力边界（经验法则）

**现阶段相对容易做好的：**

- 支架、外壳、简单减速器壳体、法兰、**参数化孔位**
- 面向 **3D 打印** 的实验件、简单子装配草图

**仍普遍吃力的：**

- **复杂人形总装**与整机可靠性设计
- **精密公差链**、产线级 **DFM**、大规模 **装配约束** 与维护性设计

## 常见误区

- **误区**：拿到 STEP 就等于仿真就绪。实际上往往需要**简化拓扑、划分碰撞凸包、对齐坐标系**，再进入 MuJoCo / Isaac 等管线。
- **误区**：把「能生成好看网格」当成「能直接上五轴」。加工可行性仍取决于**材料、公差、夹持与工序**；LLM 输出应视为**初稿**。
- **误区**：忽略合规与数据驻留。涉密或出口管制项目需单独核验供应商的**分区与合同条款**（例如 Zoo 公开站点中的 **ITAR / SOC 2** 叙事仅作线索，不构成法律意见）。
- **误区**：用对话 CAD **替代**设计评审。早期工具更应嵌入 **PR 式审图 + 测量 + 试制** 而不是替代 sign-off。

## 关联页面

- [URDF-Studio](../entities/urdf-studio.md) — Web 端机器人描述与 BOM 工作流，可与 CAD 出口衔接。
- [Articraft](../entities/articraft.md) — 可关节 3D 资产的 agentic 生成与验证回路（与 STEP/B-rep 主战场相邻、目标侧重仿真交互）。
- [Atom01 Hardware](../entities/atom01-hardware.md) — 开源硬件仓中 **CAD / BOM** 与仿真描述分层的实例。
- [Sim2Real](./sim2real.md) — 几何与动力学一致性问题。

## 推荐继续阅读

- [Zoo Text-to-CAD 教程](https://zoo.dev/docs/developer-tools/tutorials/text-to-cad)
- [Zoo ML API 文档](https://zoo.dev/docs/developer-tools/api/ml?lang=python)
- [KCL 语言文档](https://docs.zoo.dev/docs/kcl)
- [CadQuery 文档](https://cadquery.readthedocs.io/) · [Build123d 文档](https://build123d.readthedocs.io/)
- [OpenSCAD](https://openscad.org/)
- [Autodesk 生成式设计（Generative Design）](https://www.autodesk.com/solutions/generative-design)（理解「AI 增强」与「全自动整机 CAD」的差别）

## 参考来源

- [文字生成 CAD / 对话式 CAD 工具（原始资料索引）](../../sources/sites/text-to-cad-tools.md)
