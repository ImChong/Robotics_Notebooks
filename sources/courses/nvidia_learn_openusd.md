# Learn OpenUSD（NVIDIA 官方 USD 自学路径）

> 来源归档（course / NVIDIA Learn OpenUSD）

- **标题：** Learn OpenUSD
- **类型：** course
- **来源：** NVIDIA
- **原始链接：** https://docs.nvidia.com/learn-openusd/latest/index.html
- **GitHub 源码：** https://github.com/NVIDIA-Omniverse/LearnOpenUSD
- **认证：** [OpenUSD Development Professional Certification](https://www.nvidia.com/en-us/learn/certification/openusd-development-professional/)
- **入库日期：** 2026-08-30
- **一句话说明：** NVIDIA 免费开源 USD 课纲：从 stage/prim 基础到 composition arcs、资产结构、instancing 与 data exchange，面向 OpenUSD 开发认证与 Isaac/Omniverse 资产管线。

## 开源与项目页核查（步骤 2.5）

| 组件 | 开放程度 | 说明 |
|------|----------|------|
| [LearnOpenUSD](https://github.com/NVIDIA-Omniverse/LearnOpenUSD) | **已开源** | Sphinx 课纲源码；接受社区 PR |
| 在线渲染站 | **免费访问** | `docs.nvidia.com/learn-openusd/latest/` |
| OpenUSD 运行时 | **已开源** | Pixar [OpenUSD](https://openusd.org/) |
| OpenUSD Development Certification | **付费认证考试** | 课纲明确为备考路径；非代码仓 |

## 课程定位

- **免费、开源** 学习路径，帮助掌握 **OpenUSD（Universal Scene Description）** 高效 3D 工作流
- 目标：普及 USD 知识、建立 3D 管线最佳实践、支撑个人/组织 **技能验证**
- **直接对接** OpenUSD Development Certification 考试
- 课内以 **Usd Python API** + **usdview** 动手练习为主；前置要求 Python 3 基础

## 模块大纲（2026-08 站点结构）

| 顺序 | 模块 | 核心主题 |
|------|------|----------|
| 0 | [What Is OpenUSD?](https://docs.nvidia.com/learn-openusd/latest/what-openusd/index.html) | 非破坏性协作、模块化、跨 DCC 互操作 |
| 1 | [Setting the Stage](https://docs.nvidia.com/learn-openusd/latest/stage-setting/index.html) | Stage、Prim、属性/关系、路径、文件格式、元数据、time samples |
| 2 | [Scene Description Blueprints](https://docs.nvidia.com/learn-openusd/latest/scene-description-blueprints/index.html) | IsA/API schema、Scope/Xform、UsdLux 光照 |
| 3 | [Composition Basics](https://docs.nvidia.com/learn-openusd/latest/composition-basics/index.html) | Layer、LIVRPS/LIVERPS 强度序、def/over/class、reference、default prim、variant set |
| 4 | [Beyond Basics](https://docs.nvidia.com/learn-openusd/latest/beyond-basics/index.html) | primvars、Hydra、units、遍历、value resolution、model kinds |
| 5 | [Creating Composition Arcs](https://docs.nvidia.com/learn-openusd/latest/creating-composition-arcs/index.html) | sublayer、reference/payload、encapsulation、inherits/specializes、variant、relocates（LIVERPS） |
| 6 | [Asset Structure](https://docs.nvidia.com/learn-openusd/latest/asset-structure/index.html) | 资产接口、workstream/layer stack、model hierarchy、ref/payload 模式、参数化 |
| 7 | [Asset Modularity and Instancing](https://docs.nvidia.com/learn-openusd/latest/asset-modularity-instancing/index.html) | scenegraph instancing、point instancing、嵌套实例、变体 refinement |
| 8 | [Developing Data Exchange Pipelines](https://docs.nvidia.com/learn-openusd/latest/data-exchange/index.html) | 几何/材质抽取、变换、asset validation、可互操作管线 |

> **注：** Creating Composition Arcs 模块说明 OpenUSD 新增 **relocates** 弧，强度序由 LIVRPS 演进为 **LIVERPS**；站点部分页面仍混用旧缩写。

## 对 wiki 的映射

- [NVIDIA Learn OpenUSD](../../wiki/entities/nvidia-learn-openusd.md) — 本课纲编译页
- [NVIDIA Physical AI Learning](../../wiki/entities/nvidia-physical-ai-learning.md) — 门户并列路径
- [NVIDIA Omniverse](../../wiki/entities/nvidia-omniverse.md) — USD 协作仿真底座
- [Isaac Sim](../../wiki/entities/isaac-sim.md) — URDF/MJCF/CAD→USD 机器人场景
- [Blender](../../wiki/entities/blender.md) — DCC→USD 资产来源对照
