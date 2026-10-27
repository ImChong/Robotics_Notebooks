# Reducing the Memory Footprint of 3D Gaussian Splatting

- **作者：** Panagiotis Papantonakis、Georgios Kopanas、Bernhard Kerbl、Alexandre Lanvin、George Drettakis
- **机构：** Inria、Université Côte d'Azur、TU Wien
- **发表：** Proceedings of the ACM on Computer Graphics and Interactive Techniques，I3D 2024
- **项目页：** <https://repo-sam.inria.fr/fungraph/reduced_3dgs/>
- **代码：** <https://github.com/graphdeco-inria/reduced-3dgs>
- **入库日期：** 2026-10-27
- **一句话说明：** 通过高斯剪枝、球谐系数自适应分配和量化减少 3DGS 场景存储。

## 核心摘录

- 分辨率感知剪枝减少高斯基元数量；训练中途为各基元分配不同球谐阶数，减少方向辐射系数。
- 后处理以码本量化和半精度存储进一步压缩属性。项目页报告在所测标准数据集上磁盘体积缩小 27 倍、渲染速度提高 1.7 倍；结果依赖数据集和实现。
- 评估使用 Mip-NeRF 360、Deep Blending 和 Tanks and Temples；图像指标仅小幅下降。压缩外观表示并不自动提供碰撞或物理属性。

## 开源与复现

作者提供训练、压缩、渲染与评估代码，详见 [仓库归档](../repos/reduced-3dgs.md)；项目页资料见 [项目页归档](../sites/reduced-3dgs.md)。仓库 README 的数据集链接标为 TODO，复现需自行准备标准数据集。新 PLY 格式与原版不兼容，仓库提供旧格式转换脚本。

## 对 wiki 的映射

- [具身感知六种空间表征](../../wiki/concepts/embodied-perception-six-spatial-representations.md)：补充 3DGS 存储与渲染成本的工程选型。
- [原始 3DGS 论文索引](../../wiki/entities/paper-rcl-2308-04079-3d-gaussian-splatting-for-real-time-radiance-fie.md)：作为原方法的后续压缩工作。

## 参考来源

- [作者项目页](https://repo-sam.inria.fr/fungraph/reduced_3dgs/)
- [作者代码及复现说明](https://github.com/graphdeco-inria/reduced-3dgs)
