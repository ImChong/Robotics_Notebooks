# SuMa++: Efficient LiDAR-based Semantic SLAM（IROS 2019）

> 来源归档（paper）

- **标题：** SuMa++: Efficient LiDAR-based Semantic SLAM
- **类型：** paper
- **作者：** Xieyuanli Chen, Andres Milioto, Emanuele Palazzolo, Philippe Giguère, Jens Behley, Cyrill Stachniss
- **机构：** 德国波恩大学（University of Bonn）；Philippe Giguère 来自加拿大拉瓦尔大学（Laval University）（首页脚注）
- **发表：** IEEE/RSJ International Conference on Intelligent Robots and Systems（IROS）2019，pp. 4530–4537（页码取自官方仓库 README 的 BibTeX）
- **arXiv：** 未发现（论文 PDF 与官方 README 均只给出 IROS 与 IPB 主页 PDF，未给 arXiv 编号）
- **PDF：** <https://www.ipb.uni-bonn.de/wp-content/papercite-data/pdf/chen2019iros.pdf>
- **代码：** https://github.com/PRBonn/semantic_suma
- **入库日期：** 2026-11-11
- **一句话说明：** 在 SuMa（surfel 激光 SLAM）上接入 RangeNet++ 的逐点语义，用语义一致性滤除动态 surfel，并把语义相容度加入 ICP 残差权重。

## 开源状态

- **已开源**（README 核对，2026-11-11；MIT License，README 版权行 Copyright 2019）。依赖 rangenet_lib（TensorRT C++）、需自行下载预训练 darknet53 模型；README 称目前只能在 KITTI 上运行，换数据需微调或重训分割网络。

## 核对范围与通读状态

- **核对范围：** 论文 PDF 全部 8 页正文（摘要、I 引言、II 相关工作、III 方法 A–F、IV 实验 A–C、V 结论，含 Alg. 1、公式 (1)–(8)、Table I、Table II、Fig. 1–7 的图注）及参考文献列表；官方仓库 README（依赖、构建、运行说明、引用与许可）。该论文无附录或补充材料。只静态阅读，没有运行代码或模型；图像本身只通过 pdftotext 取得图注与文字，未逐图目视，数值一律取自正文与表格。未审阅仓库源码。
- **通读：** 已完成

## 核心摘录

1. **定位：** 以 SuMa [2] 为骨架，在不使用相机的前提下，仅靠 LiDAR 同时得到语义地图与更好的位姿（摘要、结论）。
2. **语义来源：** RangeNet++（SqueezeSeg 结构，DarkNet53 主干）对距离图像逐点分类，用 SemanticKITTI 标注训练；训练用 KITTI 里程计序列 00–10，序列 08 留作验证（§III-C、§IV）。
3. **精修（§III-D、Alg. 1）：** 先腐蚀（邻域内标签不一致则置空），再用顶点图深度一致性（阈值 θ）做 flood-fill，消除反投影伪影。
4. **动态滤除（§III-E，式 (1)）：** 观测语义与地图 surfel 语义不一致时，对该 surfel 的稳定性对数几率加惩罚项 odds(p_penalty)，多次观测后移除不稳定 surfel；仅处理可移动类，植被等不惩罚。
5. **语义 ICP（§III-F，式 (2)–(8)）：** 残差权重 = Huber 权重 × 语义相容度 C_semantic（标签相同取 P(y)，不同取 1−P(y)）× 稳定 surfel 指示函数。
6. **结果：** KITTI Road（Table I）SuMa 平均 1.35/6.13，SuMa++ 1.04/1.46；KITTI Odometry 训练序列（Table II）平均 SuMa 0.36/0.83，SuMa++ 0.29/0.70（旋转 deg/100m / 平移 %）；测试集平移 1.06% 对 SuMa 1.39%（§IV-B）。
7. **耗时（§IV）：** RangeNet++ 平均 75 ms/帧，surfel 建图平均 48 ms/帧，回环整合最长 190 ms；硬件 Xeon W-2123 + Quadro P4000。
8. **自述局限（§IV-C）：** 无法滤除第一帧观测中的动态物体，初始化期移除所有可移动类；植被叶片变化未处理；未来计划把语义用于回环。

## 对 wiki 的映射

- [paper-suma-plus-plus](../../wiki/entities/paper-suma-plus-plus.md)
