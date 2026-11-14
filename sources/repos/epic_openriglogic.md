---
type: source
source_type: repo
title: "EpicGames/OpenRigLogic 源码与 MetaHuman DNA/RigLogic 架构核查"
urls:
  - "https://github.com/EpicGames/OpenRigLogic"
  - "https://dev.epicgames.com/documentation/metahuman/metahuman-documentation"
created: 2026-11-14
updated: 2026-11-14
tags: [software, digital-human, character-animation, unreal-engine, epic-games]
---

# `EpicGames/OpenRigLogic` 源码与 MetaHuman 文档核查

- **官方仓库：** <https://github.com/EpicGames/OpenRigLogic>（对照 MetaHuman 官方文档）
- **核对提交：** `994deb4aca3bd455729b4719b3e2ec91d0b83401`（`main` 分支，`CMakeLists.txt` 声明 `RigLogic4 VERSION 13.5.2`）
- **核对范围：** 核查 `README.md`、`LICENSE`、`CMakeLists.txt`、`docs/design/design.md`、`include/riglogic/riglogic/{RigLogic.h,RigInstance.h}`、`include/dna/BinaryStreamReader.h`、`examples/{Basic.cpp,Advanced.cpp}` 与 Python SWIG 示例 `python/{dna,riglogic}/examples/demo.py`；并复核 MetaHuman Overview 与 Animator 文档目录及官方站点归档
- **通读：** 已完成

## 一、开源开放状态核查（步骤 2.5）

| 资源层级 | 开放状态 | 说明与限制 |
|---|---|---|
| `OpenRigLogic`（DNA + RigLogic C++ 库与 Python 绑定） | **已开源（MIT License）** | Epic Games 公开 `RigLogic4`（`v13.5.2`）与 `DNA` 读写库源码、`CMake` 构建脚本、SWIG Python 包装（`python/dna`、`python/riglogic`）与 C++/Python 示例 |
| 角色示例资产与 Faceboard Control Rig | **外部独立许可分发** | 不包含在 GitHub 代码仓内；需通过 Fab 平台单独获取（遵循 Fab Standard License） |
| MetaHuman Creator / Animator / UE 插件 | **闭源商用 / UE 生态授权** | 属于 Unreal Engine 生态工具链，生成的人物角色、材质、毛发与服装资产受 MetaHuman 许可与 Unreal Engine EULA 约束（Export > DCC Export 可导出 head/body `.dna` 与贴图，不含 Groom/Clothing） |

## 二、仓库关键发现与设计要点

1. **DNA 分层数据格式（`include/dna/`）：**
   - `BinaryStreamReader` / `JSONStreamReader` 支持按 `DataLayer`（`Descriptor` → `Definition` → `Behavior` / `Geometry` / `MachineLearnedBehavior`）及 `maxLOD` / `minLOD` 裁剪加载，降低非高模视角的加载内存。
   - 数据层涵盖关节层级、网格几何、PSD（Pose Space Deformation）、ML（MachineLearnedBehavior / Ext）、RBFBehavior 与 TwistSwingBehavior。
2. **RigLogic 4 实时求值架构（`docs/design/design.md`、`RigLogic.h`、`RigInstance.h`）：**
   - **线程安全无状态分离：** `RigLogic` 对象存储优化后的共享只读求解数据且函数线程安全；每个角色实例的输入输出缓冲区独立保存在 `RigInstance` 中。
   - **存储重构：** 相比早期 CRS 稀疏矩阵（逐行间接寻址破坏缓存与 SIMD 向量化、LOD 重复冗余以及 GeneSplicer 矩阵融合需转稠密），RigLogic 4 进行矩阵剪枝、分离 JointGroup / BlendShape（单射映射）/ MaskMultiplier（移入条件表），并采用**按 4 行分块列主序存储的 `BlockSparseMatrix`** 消除行末规约并匹配 SSE/AVX/NEON 指令集，同时让低精度 LOD 成为高精度 LOD 的严格子集以共享内存。
   - **求值顺序：** `RigLogic::calculate(instance)` 按序评估输入控制（可选 `mapGUIToRawControls`、ML 与 RBF 控制）→ PSD 组合表达 → 关节变换增量（`getJointOutputs`，每关节含平移/旋转四元数或欧拉角/缩放）→ 混合变形权重（`getBlendShapeOutputs`）→ 动画贴图权重（`getAnimatedMapOutputs`）。
3. **跨平台与可选构建优化（`CMakeLists.txt`、`README.md`）：**
   - 支持 `RL_BUILD_WITH_SSE` / `AVX` / `AVX2` / `AVX512F` / `FMA` / `NEON` 与 F16C/FP16 半精度存储（`RL_BUILD_WITH_HALF_FLOATS`）、运行时 CPU 特性检测与 SWIG Python 绑定（`RL_BUILD_PYTHON_WRAPPER`）。

## 参考链接

- [EpicGames/OpenRigLogic 仓库](https://github.com/EpicGames/OpenRigLogic)
- [MetaHuman 官方文档入口](https://dev.epicgames.com/documentation/metahuman/metahuman-documentation)
