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
- **首个公开版本：** 1.0.0，2026-10-08
- **源码开放状态：** 已开源。官方仓库公开 C++/Python API、构建与测试脚本及示例；截至入库日，项目暂不接受贡献，但欢迎提交 issue / bug report。

## 摘要

USD BRep 将 SMLib NURBS B-rep 实体建模库作为独立 C++/Python 项目公开发布。SMLib 维护拓扑实体（shell/face/edge/vertex）及其精确曲线、曲面定义；高层 SM_API 提供常用几何操作，SM_API_USD 负责与 OpenUSD 场景交换。支持以自定义 BrepArray schema 保存 B-rep，也支持标准 UsdGeomMesh 网格的导入导出与 B-rep 曲面细分。

仓库示例与文档覆盖基本体、布尔、扫描/放样、倒角、修剪曲面、网格细分、STEP 衍生文件相关示例及 USD 读写。Python 包通过 pybind11 暴露接口；运行时需要正确构建/配置 OpenUSD、Python 扩展和 omniSolid schema 插件，不能假设安装仓库即有开箱即用的 CAD 应用。

## 与 Agent 工作流的关系

该库提供可编程几何能力，不自带 Claude/Codex agent、对话式建模产品、特征树 GUI 或设计意图规划器。开发者可以在外部增加工具封装：Agent 把规格转成受约束的 API 调用，运行后检查实体有效性、尺寸/公差与质量属性，再输出 USD 或离散网格。LLM 生成结果的正确性仍需几何断言和人工审查。

## 必读材料

- [README / 构建、测试、示例与许可证](https://github.com/NVIDIA-Omniverse/usd-brep/blob/main/README.md)
- [SMLib / API 分层总览](https://github.com/NVIDIA-Omniverse/usd-brep/blob/main/.agents/docs/OVERVIEW.md)
- [USD BRep 官方文档](https://nvidia-omniverse.github.io/usd-brep/)
- [首个公开版本记录](https://github.com/NVIDIA-Omniverse/usd-brep/blob/main/CHANGELOG.md)
