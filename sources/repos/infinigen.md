# princeton-vl / infinigen

- **标题：** Infinigen — Infinite Photorealistic Worlds using Procedural Generation
- **类型：** repo
- **组织：** Princeton Vision & Learning Lab
- **链接：** <https://github.com/princeton-vl/infinigen>
- **项目页：** <https://infinigen.org/>
- **许可证：** BSD-3-Clause
- **入库日期：** 2026-10-17
- **一句话说明：** Blender/Python 程序化场景生成代码库；Infinigen V2 提供面向不同场景与资产类型的生成入口，可组合程序化几何、外观与材质，支持生成可继续编辑和渲染的场景。
- **项目页归档：** [sources/sites/infinigen.md](../sites/infinigen.md)
- **沉淀到 wiki：** [wiki/entities/infinigen.md](../../wiki/entities/infinigen.md)

## README 与项目页摘要

官方仓库将 Infinigen 定位为「使用程序化生成构建无限逼真世界」的开源项目。生成内容由规则与程序化资产构成，可通过 Python 工作流配置、组合及扩展生成器；V2 文档另列 Nature、Indoors 与 Articulated 等使用入口。工具以 Blender 场景为中心，可导出到外部 3D 文件格式。

官方列出的代表性论文包括：

- [Infinite Photorealistic Worlds using Procedural Generation](https://arxiv.org/abs/2306.09310)（CVPR 2023）
- [Infinigen Indoors: Photorealistic Indoor Scenes using Procedural Generation](https://arxiv.org/abs/2406.11824)（CVPR 2024）
- [ProcFunc: Function-Oriented Abstractions for Procedural 3D Generation in Python](https://arxiv.org/abs/2604.26943)（2026）

## 对机器人仿真的价值与边界

- 适合批量构造外观、布局与物体组合变化，为视觉策略、合成感知数据和仿真场景提供程序化输入。
- 生成的逼真外观不等于物理仿真就绪：碰撞体、质量/惯量、接触参数、任务逻辑和机器人模型仍需在下游补齐并验证。
- 官方仓库公开 BSD-3-Clause 代码；官方文档索引列有预生成数据下载说明；项目主定位仍是生成代码与可控数据构造，本次未发现单独托管的 Hugging Face 数据集仓库入口。

## 参考链接

- 项目页与开源结论：[sources/sites/infinigen.md](../sites/infinigen.md)
- Infinigen 实体页：[wiki/entities/infinigen.md](../../wiki/entities/infinigen.md)
- 官方代码：<https://github.com/princeton-vl/infinigen>
- V2 文档：<https://infinigen.cs.princeton.edu/docs/latest/Infinigen2.html>
- [Infinigen Indoors 论文](https://arxiv.org/abs/2406.11824)
