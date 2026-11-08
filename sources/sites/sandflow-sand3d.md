# Sandflow / Sand3D 官方项目页

- **类型：** site / 官方项目主页
- **名称：** Sand3D — 3D library of 100K assets open source and written as code
- **链接：** https://sandflow.ai/sand3d/
- **团队：** Sandflow
- **核查日期：** 2026-11-08
- **代码：** https://github.com/Sandflow-ai/sand3d （Apache-2.0，已公开）
- **数据集：** https://huggingface.co/collections/Sandflow/sand3d （五子集，CC BY 4.0；资产原许可保留）
- **浏览器：** https://huggingface.co/spaces/Sandflow/sand3d-explorer
- **官方技术原文：** https://huggingface.co/blog/Sandflow/sand3d-106094-3d-objects-in-one-code-substrate-for
- **API 文档：** https://github.com/Sandflow-ai/sand3d/blob/main/docs/api.md

## 项目页核验

Sandflow 官方页面将 Sand3D 定位为 **参数化 3D-as-code 资产库 + 供 LLM 读写的 Python 建模语言**。一件物体由分件函数及其尺寸、放置、材质和关节构成。来源为 Objaverse、CAD、Articraft、AutoMate、NVIDIA SimReady 五类公开资产。页面展示 10K+ 可动对象与 36K+ 关节（宣传近似数）；发布博客给出 **10,456** 可动对象的更精确值。

**开源状态：** Python 库与五类数据均已开放；作者的自动重建 agentic pipeline 尚未发布可独立复现的完整技术细节或预训练模型。商用条件请同时看代码许可证、数据集条款及每行 source_license/attribution。

## 映射

- [Sand3D 主知识节点](../../wiki/entities/sand3d.md)
- [官方源码归档](../repos/sandflow-sand3d.md)
- [官方技术文章通读归档](../blogs/sandflow_sand3d_release_2026_11_05.md)
