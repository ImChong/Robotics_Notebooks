# Reducing the Memory Footprint of 3D Gaussian Splatting

- **作者：** Panagiotis Papantonakis、Georgios Kopanas、Bernhard Kerbl、Alexandre Lanvin、George Drettakis
- **机构：** Inria、Université Côte d'Azur、TU Wien
- **发表：** Proceedings of the ACM on Computer Graphics and Interactive Techniques，I3D 2024
- **论文与项目页：** <https://repo-sam.inria.fr/fungraph/reduced_3dgs/>
- **代码：** <https://github.com/graphdeco-inria/reduced-3dgs>
- **入库日期：** 2026-10-27

## 方法与结果

- 训练时采用分辨率感知的高斯原语剪枝，减少原语数量。
- 为每个原语自适应选择球谐系数，减少方向相关外观的存储。
- 使用码本量化和半精度表示降低属性存储开销。
- 作者在测试的标准数据集上报告磁盘大小缩小 27 倍、渲染速度提升 1.7 倍；实际效果依赖场景与实现。

## 对 wiki 的映射

- [Reduced 3DGS](../../wiki/entities/paper-reduced-3dgs.md)
- [3DGS 基础论文](./3d_gaussian_splatting_arxiv_2308_04079.md)
- [NeRF 基础论文](./nerf_arxiv_2003_08934.md)

## 参考来源

- [官方项目页与论文](https://repo-sam.inria.fr/fungraph/reduced_3dgs/)
- [官方代码](https://github.com/graphdeco-inria/reduced-3dgs)
