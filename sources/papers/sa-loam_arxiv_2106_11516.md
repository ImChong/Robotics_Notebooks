# SA-LOAM: Semantic-aided LiDAR SLAM with Loop Closure（arXiv:2106.11516）

> 来源归档（paper）

- **标题：** SA-LOAM: Semantic-aided LiDAR SLAM with Loop Closure
- **作者：** Lin Li, Xin Kong, Xiangrui Zhao, Wanlong Li, Feng Wen, Hongbo Zhang, Yong Liu（浙江大学 Institute of Cyber-Systems and Control；华为诺亚方舟实验室）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2106.11516>（v1，2021-06-22，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2106.11516v1>
- **会议：** ICRA 2021（据任务说明；v1 PDF 正文未标注会议名，未独立核实）
- **代码：** 论文正文未给出自家仓库链接；检索 GitHub 时服务返回错误，**未核实**是否存在官方开源（因此 wiki 页不写 `code:`）
- **入库日期：** 2026-11-11
- **一句话说明：** 在开源 FLOAM 之上加入语义辅助 ICP（语义分组匹配、语义下采样、平面约束）与基于语义图的回环检测，在 KITTI 与 Ford Campus 上评测。

- **核对范围：** 通读 arXiv v1 PDF 全文（摘要、I 引言、II 相关工作、III 方法 A-C 含式 1-6、IV 实验含表 I-IV 与图 1-7、V 结论、参考文献列表）。该 PDF 无附录 / 补充材料。图 1-7 以 pdftotext 抽文字为主，图像内容（轨迹图、分割可视化）未逐图目视核对，只据正文图注。未下载或运行任何代码，未读其依赖的前作 [16]（arXiv:2008.11459）全文，其图匹配网络细节仅按本文转述。
- **通读：** 已完成

## 核心摘录

1. **定位（摘要 / I）：** 以 LOAM 类系统为基础，把语义同时用于里程计与回环检测；基线是开源 FLOAM [18]（不是原版 LOAM，I 引言第 4 段、IV-A）。
2. **语义辅助 ICP（III-B）：** 三点——按语义标签分组配准（受 SE-NDT [22] 启发）、按语义标签分别下采样（保留小物体）、用语义约束平面拟合（地面法向应竖直、建筑面法向应水平，不满足的平面丢弃）；残差见式 (2)，每项带语义权重 w_l。
3. **回环检测（III-C）：** 候选帧生成（式 5，距离窗口 δd = max(δmin, min(δmax, σd))，最多 Ncandi 个，超出则随机抽）→ 语义图相似度打分（前作 [16] 的图匹配网络，每帧图最多 100 节点、每节点含 xyz 与语义标签）→ 几何验证（语义辅助 ICP 对子图配准，损失大于 δr 则丢弃）→ g2o 位姿图优化（式 6）。
4. **实验设置（IV-A）：** KITTI Odometry（00-10）与 Ford Campus（01-02），均为 Velodyne HDL-64E；语义分割用在 SemanticKITTI 上预训练的 RangeNet++（19 类，保留 12 类）；i7-9750H + GTX1080；表 I 参数 α=0.1、Nr=6、Ne=20、Np=50、Nm=20、Nne=Nnp=5、σ=0.05、δmin=15、δmax=100、Ncandi=64、ζ=0.95、Nloop=5、δr=100。**语义权重 w_l 实验中取相同值**（IV-A）。
5. **KITTI 相对误差（表 II）：** 平均旋转 / 平移误差（deg/100m / %）FLOAM 0.49/1.27，Ours-ODOM 0.31/0.76，Ours-LOOP 0.28/0.76，SUMA++ 0.28/0.94，LOAM（引自原论文，仅平移）0.84%。
6. **KITTI 绝对轨迹误差（表 III，含回环序列）：** 平均 ATE（m）ISC-LOAM 2.52、SUMA 7.59、SUMA++ 2.76、Ours-ODOM 3.49、Ours-LOOP 2.34；回环相对里程计平均改善 32.9%，00 号 80.7%、05 号 75.3%。
7. **Ford Campus（表 IV）：** 平均 ATE（m）FLOAM 4.62、ISC-LOAM 6.34、SUMA 40.05、SUMA++ 33.75、Ours-ODOM 5.18、Ours-LOOP 4.16；作者称只有 Ours-LOOP 在图 6 中正确闭合回环；并承认 Ford 上语义分割明显更差（图 7）。

## 对 wiki 的映射

- [paper-sa-loam](../../wiki/entities/paper-sa-loam.md)
