# Scan Context：面向 3D 点云地图内位置识别的以自我为中心的空间描述符

> 来源归档（paper）

- **标题：** Scan Context: Egocentric Spatial Descriptor for Place Recognition within 3D Point Cloud Map
- **类型：** paper
- **作者 / 机构：** Giseop Kim、Ayoung Kim；韩国 KAIST 土木与环境工程系（Department of Civil and Environmental Engineering）
- **发表：** IROS 2018（会议信息取自任务说明；所读作者版 PDF 正文未印出会议页眉，未核实）
- **arXiv：** 无（本次检索与作者主页均未见 arXiv 版本，未核实其不存在）
- **PDF：** <https://gisbi-kim.github.io/publications/gkim-2018-iros.pdf>（作者主页，2026-11-11 下载，首页标题与作者核对一致）
- **代码：** https://github.com/irapkaist/scancontext（本次会话无法访问 GitHub，未核实仓库内容；论文正文只提到随文视频 scancontext.mp4）
- **入库日期：** 2026-11-11
- **一句话说明：** 把单帧 3D LiDAR 扫描按「方位角 × 半径」分成 60 × 20 个 bin，每个 bin 取最大高度，得到一个矩阵描述符，用列方向余弦距离加列平移搜索比较，并用旋转不变的 ring key + KD 树做两阶段检索。

## 核对范围与通读

- **核对范围：** 作者版 PDF 全文 6 页含参考文献，逐段读完摘要、引言、相关工作、第三节方法（描述符构造、相似度、两阶段搜索）、第四节实验（数据集、PR 曲线、定位精度、计算开销）、结论与参考文献；PR 曲线图（Fig. 5）只确认趋势，未逐点读数值（论文本身没有给 PR 的数值表）；未读源码与随文视频；未做复现。
- **通读：** 已完成

## 核心摘录

1. **描述符构造（III-A，式 1 至 4，Fig. 1）：** 以传感器为中心，沿方位角（0 到 2π）和半径（0 到 Lmax）等间隔划分，Ns 个 sector、Nr 个 ring；论文设 Ns = 60、Nr = 20、Lmax = 80 m（IV-A），即每 sector 6°、每 ring 4 m。bin 编码函数是 bin 内点的最大 z 值，空 bin 记 0（式 3），结果是 Nr × Ns 矩阵（式 4）。论文称这样不依赖点数，对点云密度和法向不敏感，并称其表示的是「egocentric 2.5D」信息。
2. **平移鲁棒（III-A）：** 描述符会随扫描中心位置变化（例如换车道重访），论文用 root shifting 增强：把原始点云平移到 Ntrans = 8 个邻近位置，各生成一个 scan context 一并存储；论文自述该做法「non-mandatory」并假设平移后点云相近，路口突然出现新空间等情形不成立。
3. **相似度（III-B，式 5 至 7）：** 逐列（同 sector 序号）算余弦距离再对 Ns 取平均；对候选描述符做全部 Ns 种列平移取最小值，对应 2π/Ns 分辨率下的偏航对齐；最佳平移量 n* 可作为 ICP 的偏航初值。
4. **两阶段搜索（III-C，式 8 至 10，Fig. 2、4）：** 先对每个 ring（矩阵一行）用 L0 范数占用率 ψ(ri) = ‖ri‖0 / Ns 编码成一个标量，Nr 个标量组成 ring key，旋转不变；KD 树只索引 ring key，检索 top-k 候选，再用式 6 的距离在候选中取最近且小于阈值 τ 者为回环。论文实验 k = 10 与 50（记作 scan context-10 / -50）。
5. **实验设置（IV-A，Table I）：** KITTI 00/02/05/08、NCLT 四个序列（2012-05-26、2012-08-20、2012-09-28、2013-04-05）、Complex Urban LiDAR 00/01/02/04；真值位姿距离小于 4 m 记为真阳性，排除最近 50 个相邻节点；对比 M2DP、Z-projection、ESF；CPU 为 Intel i7-6700 3.40 GHz、16 GB；scan context 与 M2DP 先用 0.6 m³ 体素下采样，Z-projection 与 ESF 用原始点云；节点数 Table I 中最多 4661（KITTI 02）。
6. **结果（IV-B，Fig. 5）：** 论文称直方图类 ESF 与 Z-projection 在所有数据集表现差；scan context-50 总体优于 scan context-10；KITTI 08 只有反向回环，scan context 明显优于其他方法，M2DP 检测不到反向回环；NCLT 因含狭窄室内路段，scan context 在低召回段精度与召回都偏低；Complex Urban 02 因窄路与重复结构对所有方法都最难，scan context 仍优于其他方法。
7. **定位（IV-C，Fig. 7、8）：** KITTI 08 上用 n* 给 point-to-point ICP 提供偏航初值，反向回环无初值时配准易失败；论文给出有 / 无初值的耗时与 RMSE 曲线（图，无数值表）。
8. **耗时（IV-D，Table II，KITTI 00，Matlab）：** 描述符计算 scan context 0.1291 s（含 8 个平移增强，论文称不含增强时单个仅 0.0143 s），M2DP 0.0218 s、Z-projection 0.0472 s、ESF 0.0635 s；回环搜索 scan context-10 为 0.0807 s、scan context-50 为 0.3331 s，M2DP 0.0032 s、Z-projection 0.0035 s、ESF 0.0043 s。论文自述 scan context 搜索时间比其他全局描述符长，但在合理范围（Matlab 上 2 到 5 Hz）。
9. **论文自述局限与展望（IV-B、V）：** 垂直高度变化小的室内环境表现受限；重复结构场景（Complex Urban）较难；展望是加入其他 bin 编码层（例如语义）。

## 对 wiki 的映射

- [paper-scan-context](../../wiki/entities/paper-scan-context.md)
