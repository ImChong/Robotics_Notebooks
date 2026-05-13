---
type: concept
tags: [cad, generative-ai, hardware, design, api, manufacturing]
status: complete
date: 2026-05-13
updated: 2026-05-13
related:
  - ../entities/urdf-studio.md
  - ../entities/atom01-hardware.md
  - ./sim2real.md
sources:
  - ../../sources/sites/text-to-cad-tools.md
summary: "文字生成 CAD 指用自然语言或对话驱动可编辑的制造向几何（常见 B-rep / STEP），与纯三角网格生成在机器人硬件与仿真链路中的用途不同。"
---

# 文字生成 CAD（Text-to-CAD）

**文字生成 CAD** 指用**自然语言提示**或**对话式代理**，自动生成或迭代**可编辑的 CAD 几何**（常见为 **B-rep** 实体），并通常导出 **STEP** 等交换格式；其工程目标偏向**加工、公差与装配**，而不是仅输出可视化网格。

## 为什么对机器人研究重要

- **硬件与夹具链路**：从概念支架、传感器支架到末端执行器零件，研发流程仍大量依赖 **STEP / 参数化模型** 与供应链沟通；若 LLM 参与设计，需要知道「生成的是否为制造级模型」。
- **与仿真资产的关系**：仿真常用的 **URDF / MJCF** 多消费 **网格与惯性近似**；STEP 往往是**上游真值**或**加工真值**，中间通常还有 **CAD → 网格 / URDF** 的转换与简化（见 [URDF-Studio](../entities/urdf-studio.md) 等工具链）。
- **Sim2Real 提示**：仿真里碰撞几何若与加工件不一致，接触与质量属性会漂移；文字生成 CAD 若用于零件原型，仍要做**独立几何校验与测量**（参见 [Sim2Real](./sim2real.md) 中关于模型一致性的讨论）。

## 能力分野（选型心智模型）

| 维度 | 制造向 Text-to-CAD | 常见 Text-to-3D（网格） |
|------|---------------------|-------------------------|
| 典型输出 | STEP、带特征或脚本的参数模型 | OBJ、STL、GLB 等 |
| 可编辑性 | 强调尺寸、约束与工程修改 | 多为网格编辑 / 重拓扑 |
| 与 CNC / 公差分析 | 厂商常以此为主叙事 | 通常较弱或非目标 |
| 与游戏 / 渲染 | 可选 | 主战场 |

## 代表产品（简表）

详细链接与原始摘录见 [sources/sites/text-to-cad-tools.md](../../sources/sites/text-to-cad-tools.md)。

- **Zoo（KittyCAD）**：桌面 **Zoo Design Studio** + 对话代理 **Zookeeper**；公开 **Text-to-CAD API** 与 **`kittycad` Python 包**；教程说明默认返回 **GLTF 与 STEP**，并强调提示词需清楚描述零件；几何与脚本层与 **KCL** 语言文档互链。
- **Maket.ai**：偏 **住宅平面 / AEC**，强调 **DWG / DXF** 与建筑工作流，与通用机械零件 CAD 场景不同。
- **GrandpaCAD / PartWork AI**：公开材料宣称 **API + STEP / B-rep** 路线；具体 SLA 与格式细节以官网为准。
- **ModelsLab 等 Text-to-3D API**：偏 **网格格式**，适合可视化或部分 3D 打印，**不**应默认等同于完整 STEP 特征树。

## 常见误区

- **误区**：拿到 STEP 就等于仿真就绪。实际上往往需要**简化拓扑、划分碰撞凸包、对齐坐标系**，再进入 MuJoCo / Isaac 等管线。
- **误区**：把「能生成好看网格」当成「能直接上五轴」。加工可行性仍取决于**材料、公差、夹持与工序**；LLM 输出应视为**初稿**。
- **误区**：忽略合规与数据驻留。涉密或出口管制项目需单独核验供应商的**分区与合同条款**（例如 Zoo 公开站点中的 **ITAR / SOC 2** 叙事仅作线索，不构成法律意见）。

## 关联页面

- [URDF-Studio](../entities/urdf-studio.md) — Web 端机器人描述与 BOM 工作流，可与 CAD 出口衔接。
- [Atom01 Hardware](../entities/atom01-hardware.md) — 开源硬件仓中 **CAD / BOM** 与仿真描述分层的实例。
- [Sim2Real](./sim2real.md) — 几何与动力学一致性问题。

## 推荐继续阅读

- [Zoo Text-to-CAD 教程](https://zoo.dev/docs/developer-tools/tutorials/text-to-cad)（官方：Python 调用与轮询模式）
- [Zoo ML API 文档](https://zoo.dev/docs/developer-tools/api/ml?lang=python)
- [KCL 语言文档](https://docs.zoo.dev/docs/kcl)

## 参考来源

- [文字生成 CAD / 对话式 CAD 工具（原始资料索引）](../../sources/sites/text-to-cad-tools.md)
