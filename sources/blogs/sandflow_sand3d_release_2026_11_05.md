# Sand3D: 106,094 3D objects in one code substrate for language models

- **类型：** 官方技术发布博客 / 一手原文
- **作者：** Thibault Neveu（Sandflow）
- **发布日期：** 2026-11-05
- **原文：** https://huggingface.co/blog/Sandflow/sand3d-106094-3d-objects-in-one-code-substrate-for
- **官网：** https://sandflow.ai/sand3d/
- **代码：** https://github.com/Sandflow-ai/sand3d
- **数据：** https://huggingface.co/collections/Sandflow/sand3d
- **核对范围：** 已通读原始发布文章的完整正文，包括 A library written for models、Structure first, details after、A recipe for the object、Clean by construction、How the data was made、Limitations、Licence、Links；并对照五份官方 HF 数据卡及 GitHub README。
- **通读：** 已完成
- **入库日期：** 2026-11-08

## 作者声明的事实（不等于独立复测）

- **106,094** 个独立 3D 对象；**10,456** 个为 articulated。
- 五子集：Objaverse **66,887**；CAD **24,495**（Zero-To-CAD **16,766** + CADFS **7,729**）；Articraft **9,209**；AutoMate **3,169**（872 assemblies + 2,297 parts）；NVIDIA SimReady **2,334**。总和 106,094。
- 每个对象包含可读的 sand3d Python 程序、程序导出的 OpenUSD 文件和渲染图；每一部件可命名、参数化，关节可声明轴及限位。
- 为支持从左至右编写程序，作者提出 **architecture-first / structure-first**：入口函数先列出部件与装配调用，各部件函数之后定义；也允许 attach 追加可独立删除的细节。
- **3.8× token 对比**只来自作者在 **30 个 CAD 零件**上，将人工编写 sand3d 与 LLM 编写 Blender Python 的例子做的 token 比较；不能外推为所有 3D 任务的通用模型加速倍率。
- 数据由 Sandflow 尚未完整公开细节的 agentic iterative reconstruction pipeline 对原始资产重写，再用 sand3d 导出；几何 F-score 以 bounding-box diagonal 的 1% 为阈值计算。筛选门槛：Objaverse 与静态 SimReady 0.70；CAD 与刚性 Articraft 0.90；可关节对象要求各部件 0.90 且关节正确（作者口径）。
- 当前主要覆盖硬表面形状；人物、动物和有机物体仍为未来目标；细节可能因重建而丢失。
- 数据集发布 CC BY 4.0，**每件资产保留其来源许可证**；Python 库 Apache-2.0，允许商用但使用者需遵守各层许可。

## 对知识库的映射

- [Sand3D 实体节点](../../wiki/entities/sand3d.md)
- [官方站点归档](../sites/sandflow-sand3d.md)
- [官方仓库源码归档](../repos/sandflow-sand3d.md)

## 证据边界

2026-11-08 只对照公开作者资料、数据卡、源码，**未**重跑重建代理、逐条验证 106,094 个资产，也未执行 Blender/URDF/Isaac Sim 实测；官方表示更详细的重建管线报告尚待发布。
