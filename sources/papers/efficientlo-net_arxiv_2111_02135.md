# EfficientLO-Net：Efficient 3D Deep LiDAR Odometry（arXiv:2111.02135）

> 来源归档（paper）

- **标题：** Efficient 3D Deep LiDAR Odometry（v1 内网络名 PWCLO-Net，v2 起改名 EfficientLO-Net）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2111.02135>
- **PDF：** <https://arxiv.org/pdf/2111.02135v2>
- **代码：** https://github.com/IRMVLab/EfficientLO-Net
- **作者 / 机构：** Guangming Wang、Xinrui Wu、Shuyang Jiang、Zhe Liu、Hesheng Wang（上海交通大学自动化系 / 计算机系；Zhe Liu 为剑桥大学）
- **发表：** IEEE TPAMI（arXiv 备注 Accepted by PAMI 2022；仓库 README 给出 TPAMI 第 45 卷第 5 期 5749-5765，DOI 10.1109/TPAMI.2022.3207015）
- **入库日期：** 2026-11-11
- **一句话说明：** 把 LiDAR 点云按圆柱投影排成有序 XYZ 坐标图，仍在原始 3D 坐标上做特征学习，用 PWC（金字塔 + 位姿 warp + 注意力 cost volume）+ 可训练嵌入掩码从粗到细回归位姿，纯里程计（无建图）。
- **核对范围：** arXiv v1（2021-11-03，网络名 PWCLO-Net，仅 KITTI）正文 §1-§6 与表 1-8、图 1-13 文字说明逐段读完；v2（最终版，网络名 EfficientLO-Net，加入 KITTI 11-21、M2DGR、Argoverse 与 A-LOAM 对照）读了摘要、贡献列表、§5.1 表 2/7/8/9/10 数值、§5.5 表 11/12、结论，其余与 v1 相同部分未逐字复读；无附录 / 补充材料可读；参考文献仅查 [39]；仓库 README 读了前部（简介、引用、依赖、使用步骤开头），未读完，源码未读；图只读了图注，未看图像本身。
- **通读：** 已完成

## 版本差异（核对「第一个」「优于 LOAM」时必须区分）

| 项 | v1（2021-11） | v2（2022-09，最终版） |
|----|----------------|------------------------|
| 网络名 | PWCLO-Net（摘要、贡献） | EfficientLO-Net（摘要、贡献、README 标题） |
| 数据集 | 仅 KITTI odometry | KITTI + M2DGR + Argoverse（摘要） |
| LOAM 对照 | LOAM 数值取自 LO-Net 论文（表 2 注：除最后三行外数据来自 [13]），Full LOAM 07-10 均值 trel 1.145 | 表 2 新增 Full LOAM 0.828 与 Full A-LOAM 0.810（A-LOAM 为自跑），表 2 把 PWCLO-Net [39] 作为单独一行 |
| 摘要措辞 | 「outperforms all recent learning-based methods and even … LOAM with mapping optimization, on most sequences of KITTI odometry dataset」 | 同句保留 |

注：v2 的表 8（运行时间）里 Full LOAM 07-10 均值仍写 1.145 / 0.498，与同篇表 2 的 0.828 不一致（作者未解释，属论文内部不一致）。

## 核心摘录

1. **「第一个」的原文措辞（v2 §6 结论）：** “To the best of our knowledge, our method is the first efficient 3D deep LiDAR odometry”，即「第一个高效的 3D 深度 LiDAR 里程计」，带「据我们所知」限定，主语是「高效」+「直接在 3D 点上学习」，不是「第一个端到端」。v2 贡献 1 写 “a totally end-to-end efficient framework … all modules are fully differentiable”（「完全端到端」是作者对自己框架的描述），摘要写 “An efficient 3D point cloud learning architecture … is first proposed”。§2.3 另说 “a PWC structure for LiDAR odometry is built for the first time”。论文并未声称 LO-Net 等基于投影的网络不是端到端；LO-Net [13] 在 §1 被描述为另训练一个掩码网络。
2. **输入表示（§3.1）：** 公式 (1)(2) 做圆柱投影得到 H×W×3 的 XYZ 坐标图（H=64、W=1800，§4.2），投影只用于组织数据，特征学习仍在原始 3D 坐标上（§3.1）；仅使用点坐标，不用反射率（§4.1）；车周 30×30 m² 以外的点被滤除（§4.1）。
3. **投影感知 3D 特征学习（§3.2）：** 步长采样（SBS）、在投影平面固定核内分组、3D 距离过滤后随机选 K 点、MLP+max 聚合。表 8(a)/(b)：全部点输入 + 投影感知，单帧总耗时 45.3 ms（数据准备 8.3 ms + 推理 37.0 ms，单张 Titan RTX，KITTI 00）；§1 称可达 20 Hz。
4. **PWC 结构（§3.3-§3.5）：** 4 层 set conv 金字塔，注意力 cost volume 放在倒数第二层（表 9(d)），之后 3 次 pose warp-refinement；位姿用四元数 + 平移，每层输出参与多尺度监督（公式 (11)(16)(17)，α=1.6/0.8/0.4，L=4，§4.2）。完全监督训练，不是自监督。
5. **嵌入掩码（§3.4、§5.2、§5.3）：** 由 cost volume 嵌入特征经共享 MLP + softmax 得到逐点权重，并随层级细化；表 9(a) 去掉掩码均值 trel 由 0.795 变为 1.218（去掉掩码优化 0.898）；图 12 显示建筑、钢筋、静止车权重高，行驶车、骑车人、灌木杂草权重低（作者定性可视化）。
6. **KITTI 主结果（表 2，v2，训练 00-06、测试 07-10，指标 trel % / rrel °/100 m）：** 本方法 07-10 均值 0.795 / 0.395；Full LOAM 0.828 / 0.498；Full A-LOAM 0.810 / 0.323；LO-Net 1.748 / 0.793；DMLO 1.008 / 0.538；PWCLO-Net [39] 1.085 / 0.490；A-LOAM w/o mapping 3.894 / 1.641。
   - 逐序列（trel）：07：本方法 0.46 vs Full LOAM 0.63；08：1.14 vs 1.12；09：0.78 vs 0.77；10：0.80 vs 0.79。即本方法只在 07 明显领先，08-10 与 Full LOAM 基本持平且略差；均值领先主要由 07 拉动（本条为读表后的计算，非作者原话）。
   - v1 表 2 中 Full LOAM 为 1.145（取自 LO-Net 论文），本方法 0.795，领先幅度明显更大；v2 换成作者重新给出的 LOAM 数值后领先收窄。
7. **其它 KITTI 划分（表 4/5/6）：** 与 LodoNet [20]（训练 00-06,09-10，测 07-08）、DeepPCO [12]（测 04、10）、Cho 等无监督 [43]（训练 00-08，测 09-10）比较，均优于对方（表 4：07-08 均值 trel / rrel 本方法 0.875 / 0.455，LodoNet 1.950 / 1.305（由逐序列数值核算一致）；表 5：0.820 vs 2.550；表 6：0.900 vs 4.945，数值取自表末均值栏）。
8. **KITTI 官方测试序列 11-21（v2 表 7，无建图）：** 本方法均值 trel 1.92 / rrel 0.52，对比 pyLiDAR 2.91 / 0.80、SuMa 4.91 / 1.53、A-LOAM w/o mapping 9.37 / 1.92、LeGO-LOAM w/o mapping 19.57 / 4.93；作者说明该设置下先在 00-06 训练再用 07-10 微调（§5.1）。对照方法均为「无建图」版本。
9. **M2DGR（v2 表 11，仅 ATE，m，VLP-32C）：** Outdoor01 本方法 1.030（最好）、Hall05 2.534（pyLiDAR 2.209 更好）、Room-dark06 0.314（最好）、Lift04 1.741（pyLiDAR 1.504 更好，SuMa 失败）、均值 本方法 1.405 vs pyLiDAR 1.335。作者自述 “similar performance to pyLiDAR”（§5.5），并未说全面优于。
10. **Argoverse（v2 表 12，序列 00-23，两个 VLP-32 激光雷达）：** ATE 本方法 0.751 vs pyLiDAR 6.900 / LeGO-LOAM 4.537 / A-LOAM 4.138 / SuMa 3.663；RPE 0.018 vs 0.109 / 0.110 / 0.066 / 0.039；对照均为无建图。
11. **作者承认的局限 / 未来工作（§6）：** 只做里程计，未与建图优化结合，是「值得探索的方向」；§5.4 图 13(d) 为失败案例（大量动态物体、少量静态刚体时有小误差）；v1 称本文代码“will be released soon”，v2 摘要称已开源。
12. **消融（表 9、表 10）：** 去 cost volume 的均值 trel 11.790；换成 FlowNet3D 嵌入 0.870、点 cost volume 1.048；去 pose warp-refinement 3.76，只去 warp 3.145；去掉地面点反而更差（保留地面 0.795，见 §5.2「Removing or Reserving Ground」）；随机选点优于 KNN（§5.2）。

## 对 wiki 的映射

- [paper-efficientlo-net](../../wiki/entities/paper-efficientlo-net.md)
