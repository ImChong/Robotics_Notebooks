# COLMAP — 一手资料归档

> 来源层（官方代码/仓库）；视觉与三维重建一手资料专题，2026-11-08。

- **标题：** COLMAP
- **类型：** repo
- **链接：** https://github.com/colmap/colmap
- **入库日期：** 2026-11-08
- **一句话说明：** SfM 与多视图立体重建系统。
- **核对范围：** 已完整阅读官方 README 和 doc/tutorial.rst（包含 SfM、MVS、相机/rig/frame、采集建议、特征匹配、密集融合和失败边界），核对 doc/cli.rst 的命令及 src/colmap/exe/colmap.cc 的 CLI 分发入口；未执行程序。
- **通读：** 已完成

## 直接一手来源（同一项目不同入口）

- [COLMAP 源码](https://github.com/colmap/colmap) **（本归档主类型）**

## 完整使用资料与主要源码归纳

1. **相机与观测**：照片读取时可从 EXIF 获取焦距初值，也可输入手工标定内参；相同物理相机、焦距和分辨率的图像可共享 camera model，rig/frame 描述多相机同步采集。
2. **特征与匹配**：默认 SIFT 或支持的学习特征提取；匹配方式可选 exhaustive、sequential、vocabulary tree、spatial、transitive；几何验证剔除错误像素匹配。
3. **增量式 SfM**：从初始图像对建立场景，不断注册新图像、三角化稀疏点并进行局部/全局 BA；未注册图像可能需要提高匹配覆盖或改变初始化图像对。
4. **稠密 MVS**：稀疏模型先去畸变，再运行 PatchMatch stereo 获取深度与法线，融合成 `fused.ply`，最后用 Poisson / Delaunay 生成网格。
5. **关键输出**：新版稀疏路径中的 `rigs.bin`、`cameras.bin`、`frames.bin`、`images.bin`、`points3D.bin`；数据库 `database.db` 为 SQLite。

## 运行边界和工程复现

最简命令：`colmap automatic_reconstructor --workspace_path path/to/project --image_path path/to/project/images`；可拆成 `feature_extractor`、`exhaustive_matcher`、`mapper`、`image_undistorter`、`patch_match_stereo`、`stereo_fusion`。2026 官方文档还提到 `global_mapper` 及 `view_graph_calibrator` 备选方案；全局 SfM 在没有准确焦距先验时更需标定。Python 可用 `pycolmap`，必须核对 CUDA/CPU 依赖和大批图像计算开销。

失败包括少纹理/重复纹理、曝光变化、纯旋转导致三角化几何退化、错误畸变模型、内参固定错、MVS 显存不足。匹配全部图像随 N 约二次增长，不适合上万图无筛选。SfM 是离线渐进重建，不保证随传感器时钟在线同步处理；把 COLMAP 生成的相机位姿用于 3DGS 训练时必须保留坐标变换约定。

## 方法与边界

SfM 与多视图立体重建系统。本档案定位官方**开源软件项目**，已审阅主 README、技术教程关键流程及源码入口；未把未运行结果标为实测。

## 对 wiki 的映射与待核验事项

- 来源汇总：[视觉三维专题 90 条来源索引](../sites/vision3d_research_primary_sources_2026_11_08.md)。
- 在升格独立 wiki 实体页前，先按论文 ID / 项目页 / 源码查重，避免与既有节点及不同项目同仓实现冲突。
- [x] 本软件项目官方 README、官方教程与核心使用流程已阅读，保留真实工程限制。
- [x] 已核对官方 CLI 或源码样例的实际调用入口，时序图见独立实体。
- [x] 已新增 [COLMAP](../../wiki/entities/colmap.md)，健康度由 CI/规则复核。
