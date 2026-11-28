---
title: "NVIDIA USD BRep / SMLib"
type: repo
source: https://github.com/NVIDIA-Omniverse/usd-brep
date: 2026-10-13
project: https://nvidia-omniverse.github.io/usd-brep/
code: https://github.com/NVIDIA-Omniverse/usd-brep
license: Apache-2.0
---

# NVIDIA USD BRep / SMLib

- **项目：** [USD BRep 官方文档](https://nvidia-omniverse.github.io/usd-brep/)
- **代码：** [NVIDIA-Omniverse/usd-brep](https://github.com/NVIDIA-Omniverse/usd-brep)
- **类型：** NURBS 边界表示（B-rep）实体建模内核与 OpenUSD 互操作库
- **许可证：** Apache-2.0；仓库中的 `repo.sh`、`repo.bat` 和 repoman bootstrapper 使用 MIT，第三方依赖另见 `THIRD_PARTY_NOTICES.md`
- **首个公开版本：** 1.0.0（2026-10-08），最新核对版本 **1.0.2（2026-11-18）**
- **入库日期：** 2026-10-13
- **通读日期：** 2026-11-28
- **核对范围：** 通读 `NVIDIA-Omniverse/usd-brep`（`main` 分支）`README.md`、`.agents/docs/OVERVIEW.md`、`CHANGELOG.md`（1.0.1–1.0.2）、`tools/pyproject/README.md`、`Examples/README.md`、`source/SM_API_USD/inc/SmApiUsd.h` 与 `tools/scripts/example_generate_asset.py`。
- **通读：** 已完成
- **源码开放状态：** 已开源。官方仓库公开 C++/Python API、构建与测试脚本及示例；项目发布 PyPI 二进制轮子 `pip install usd-brep`（CPython 3.12 Linux x86_64/aarch64 与 Windows）；截至入库与复核日暂不接受外部 PR 贡献，但欢迎提交 issue / bug report。

## 一手源码与规格核对要点（2026-11-28）

1. **五层架构与接口职责（`.agents/docs/OVERVIEW.md` & `README.md`）：**
   - `SMLib`（`source/SMLib/`）：底层 C++ 实体建模内核，管理 `SmBrep` / `SmShell` / `SmFace` / `SmEdge` / `SmVertex` 与 NURBS / 解析曲面（Plane/Cylinder/Cone/Sphere/Torus；线/圆/椭圆解析边）。
   - `SM_API`（`source/SM_API/`）：稳定 C 风格高层 C++ API（`SmApiPrimitives.h`、`SmApiBrep.h`、`SmApiFillet...`、`SmApiHeal.h`、`SmTess.h`），使用懒初始化进程全局 `SmContext`。
   - `BREP_USD_DATA` / `BREP_SM_USD`：定义 `omniSolid` Schema（`BrepArray` prim）及内部 `SmBrep ↔ USD` 序列化变换。
   - `SM_API_USD`（`SmApiUsd.h`）：USD 文件往返与批量场景工具，包括 `SmApiUsdEnsurePluginRegistered()`、`SmApiUsdImportBreps/ExportBreps`、`SmApiUsdTessellateFile()`、`SmApiUsdHealFile()`（1.0.1 新增，保留 layers/references/instancing 仅修复单层定义的 B-rep）、`SmApiUsdValidateGeometryFile()`（1.0.2 新增，并行执行 `SmBrep::AssertValid`）。
   - `SmPyLib`（`_omni_solid` / `usd_brep`）：pybind11 Python 绑定；将几何输出打包为 `Brep` 与离散 `PolyBrep`（支持质量属性、封闭体检查、射线检测与 `to_mesh_arrays`）。
2. **变更语义契约与安全约束（`OVERVIEW.md` & `SmApiUsd.h`）：**
   - 公开 Python 接口严格区分三类所有权语义：`Pure`（`create_box`、`tessellate` 返回新对象）、`Consumes`（`boolean_union`、`boolean_difference` 消耗输入所有权）、`Mutates`（`fillet_edges`、`cut`、`heal_brep` 原地修改并返回句柄，需先 `.copy()` 保护原件）。
   - `SmApiUsdExportBrep(s)` 传入非 `const` 指针并会向 `SmBrep` 附加 UV 裁剪曲线/目标 SdfPath 属性与几何缓存，**不可**在同一 `SmBrep` 上与其他操作并发调用。
3. **PyPI 发行包与验证兼容性（`tools/pyproject/README.md` & `CHANGELOG.md` 1.0.2）：**
   - `usd-brep` 轮子内置了定制编译的 **OpenUSD 25.11** 运行时（不向 `pxr` 暴露对象，靠文件路径交换 Stage）；可与不同版本的 `usd-core`（如 Linux 下 26.8）共存，但**严禁与 PyPI 同版本 `usd-core==25.11` 混装在同一进程**，否则动态链接符号冲突导致崩溃。
   - CLI 命令包括 `usd-brep-heal`、`usd-brep-validate-geometry` 与 `nvidia_usd_validate`（`pip install "usd-brep[validation]"` 注册 `BrepUsdSchemaValidator`，注意 Linux aarch64 无 `usd-core` 轮子故不可用此 extra）。
4. **端到端资产管线脚本核对（`tools/scripts/example_generate_asset.py`）：**
   - 典型运行路径：`view_usd._ensure_usd_runtime_env()` 配置动态库 → `_omni_solid` 构建基本体并倒角 → `sm.tessellate()` → `polybrep.to_mesh_arrays(face_varying_normals=True)` → `pxr.Usd/UsdGeom.Mesh` 写入纯网格 `.usda`（带 `displayColor`）→ 可选调起 `tools/scripts/view_usd.py --screenshot`（另有 `hdUsdBrep` Hydra 插件、OpenCASCADE 双向转换 `occt_to_usd`/`usd_to_occt` 及 `smlib_gui` 桌面检查器）。

USD BRep 将 SMLib NURBS B-rep 实体建模库作为独立 C++/Python 项目公开发布。SMLib 维护拓扑实体（shell/face/edge/vertex）及其精确曲线、曲面定义；高层 SM_API 提供常用几何操作，SM_API_USD 负责与 OpenUSD 场景交换。支持以自定义 BrepArray schema 保存 B-rep，也支持标准 UsdGeomMesh 网格的导入导出与 B-rep 曲面细分。

仓库示例与文档覆盖基本体、布尔、扫描/放样、倒角、修剪曲面、网格细分、STEP 衍生文件相关示例及 USD 读写。Python 包通过 pybind11 暴露接口；运行时需要正确构建/配置 OpenUSD、Python 扩展和 omniSolid schema 插件，不能假设安装仓库即有开箱即用的 CAD 应用。

## 与 Agent 工作流的关系

该库提供可编程几何能力，不自带 Claude/Codex agent、对话式建模产品、特征树 GUI 或设计意图规划器。开发者可以在外部增加工具封装：Agent 把规格转成受约束的 API 调用，运行后检查实体有效性、尺寸/公差与质量属性，再输出 USD 或离散网格。LLM 生成结果的正确性仍需几何断言和人工审查。

## 必读材料

- [README / 构建、测试、示例与许可证](https://github.com/NVIDIA-Omniverse/usd-brep/blob/main/README.md)
- [SMLib / API 分层总览](https://github.com/NVIDIA-Omniverse/usd-brep/blob/main/.agents/docs/OVERVIEW.md)
- [USD BRep 官方文档](https://nvidia-omniverse.github.io/usd-brep/)
- [首个公开版本记录](https://github.com/NVIDIA-Omniverse/usd-brep/blob/main/CHANGELOG.md)
