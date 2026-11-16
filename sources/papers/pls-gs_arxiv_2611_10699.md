# PLS-GS（arXiv:2611.10699）

> 来源归档（paper）

- **标题：** PLS-GS: Probabilistic Lidar Scaffolds for Compact, Surface-Aligned Gaussian Splatting
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2611.10699>
- **PDF：** <https://arxiv.org/pdf/2611.10699v1>（v1，2026-11-07 提交，cs.CV；首页标注为 WACV 2027 预印本，终稿将见 IEEE Xplore）
- **作者与机构：** Krishnan Harikumar、Alejandro Fontan、Thierry Peynot、Emili Hernández、Niko Sünderhauf；QUT Centre for Robotics（昆士兰科技大学）、Emesent
- **代码：** <https://github.com/k-harikumar/PLS-GS>（README 读于 2026-11-11：代码「Release is in preparation and coming soon」，未发布）
- **入库日期：** 2026-11-11
- **一句话说明：** 在相机加高密度 LiDAR 的场景里，用 6 维（位置加颜色）层级高斯混合（HGMM）当 Scaffold-GS 的锚点支架，在各分量的特征向量局部曲面坐标系里解码神经高斯，再加马氏距离支撑损失，得到更紧凑、更贴近 LiDAR 表面的 3DGS。
- **核对范围：** 读了 arXiv v1 PDF 全部 14 页：正文第 1–8 章（引言、相关工作、预备知识、方法 4.1–4.4、实验设置 5.1–5.4、结果 6.1–6.4、局限、结论）、参考文献，以及文末补充材料 S-I（基线配置、支架构造、训练、几何评测协议）与 S-II（定性对比、支撑损失权重扫描 Table S2、每锚点神经高斯数 Table S3）。图以图注与文字读，表格数字取自 pdftotext 抽取文本并与正文引用交叉核对；未逐张目视检查图像。仓库只读了 README（未取得文件树，未运行任何代码）；项目页 k-harikumar.github.io/PLS-GS 未访问，未核实。
- **通读：** 已完成

## 开源状态

- 论文摘要写 `Code is available at https://github.com/kharikumar/PLS-GS`（无连字符，作者名写法与实际仓库 `k-harikumar` 不同）。
- 实际仓库 `k-harikumar/PLS-GS` 的 README 写明代码发布「in preparation and coming soon」，并说明实现基于 Scaffold-GS。2026-11-11 读取时**尚未开源代码**。

## 核心摘录（均为论文所述，章节见括号）

1. **输入假设：** 彩色点云（LiDAR 点加 SfM 点），LiDAR 须已与 COLMAP 坐标系配准；作者称 GauU-Scene V2 是其所知唯一满足该条件的大尺度公开基准（Sec. 5.1）。
2. **方法：** 位置加颜色 6 维标准化后拟合 HGMM（每节点 8 个子分量，叶层级 7，EM 至多 20 次迭代），每个叶分量一个锚点，锚点位置取均值并固定，协方差特征分解给出曲面坐标系 U 与初始尺度；神经高斯中心 μ = x + U(o ⊙ s)（Eq. 8–10，Sec. 4.1–4.2）。
3. **损失：** L = L_photo + λs·L_support，支撑损失为各神经高斯在锚点局部系下的马氏距离平方均值，λs = 1e-3（Eq. 11–12，Sec. 4.3，5.3）。
4. **生长：** 对高光度残差的神经高斯拟合局部 6D HGMM，叶分量作为新锚点；补充材料给出生长窗口为第 1500 至 15000 次迭代、每 100 次一次（S-I.3）。
5. **数据与设置：** GauU-Scene V2 的 5 个场景（Campus、Village、College、Modern、Russian），图像长边 800 像素，每 8 张留 1 张测试；Adam 训练 30K 迭代，单张 24 GB GPU；基线为 Scaffold-GS（SfM 初始化与 LiDAR 初始化）（Sec. 5.1–5.3）。
6. **主结果（Table 1 均值）：** 精度 0.330（LiDAR 基线）到 0.484，Chamfer 0.820 到 0.703 m，锚点终值 1308K 到 460K（2.8 倍少），模型 413.2 到 172.9 MB（42%），FPS 114.4 到 190.4（1.7 倍），PSNR 落后 1.03（对 LiDAR 基线）至 1.65 dB（对 SfM 基线）；召回与 F1 均值上基线更高（Sec. 6.1）。
7. **匹配容量（Table 2）：** 以 PLS-GS 最终锚点集初始化 Scaffold-GS 且关闭生长，PLS-GS 精度 1.54 倍、召回 1.40 倍、F1 1.45 倍、Chamfer 降 40%；PSNR 差距收窄到 0.21 dB；FPS 反而较低（190 对 272）（Sec. 6.2）。
8. **消融（Table 3，College）：** 仅加曲面坐标系不改善放置，加支撑损失后精度升 2.27 倍、Chamfer 降 2.95 倍；λs 从 1e-4 到 1e-1 精度单调上升，但 1e-1 时 PSNR 掉到 23.85 dB；概率生长只加 1.06 倍容量，光度生长加 3.23 倍（Sec. 6.3）。
9. **作者承认的局限：** 渲染质量仍低于使用无约束光度生长的 Scaffold-GS（Sec. 7）。

## 对 wiki 的映射

- [paper-pls-gs](../../wiki/entities/paper-pls-gs.md)
