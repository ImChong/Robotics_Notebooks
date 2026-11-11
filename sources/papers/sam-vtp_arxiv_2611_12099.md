# SAM-VTP（arXiv:2611.12099）

> 来源归档（paper）

- **标题：** SAM-VTP: Video Memory and Measured Geometry for Continuous Traversability Segmentation
- **类型：** paper
- **作者 / 机构：** Lucas Hoefs, Hui Xie, Senjian An, Fangyi Zhang；Curtin University（School of EECMS，Bentley, WA）
- **arXiv：** <https://arxiv.org/abs/2611.12099>（v1，2026-11-09，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.12099v1>
- **代码：** https://github.com/lucashoefs/sam_vtp （部署栈：https://github.com/lucashoefs/sam_vtp_jetacker，论文 Sec. I 给出）
- **入库日期：** 2026-11-11
- **一句话说明：** 微调 SAM2 视频记忆通路做无提示的逐像素连续可通行性分割，并把深度测得的坡度 / 粗糙度 / 高度融合进图像特征；TartanGround 合成数据评测加 JetAcker 实机闭环。
- **核对范围：** 读 arXiv v1 PDF 全文（摘要、Sec. I-V、Fig. 1-5、Table I-III、参考文献列表）；该版本无附录 / 补充材料。项目页：论文未给独立项目页。仓库：静态读 README、LICENSE、docs/labels.md（clone 深度 1，commit be669c7，未运行代码）；Hugging Face 权重、sam_vtp_jetacker 仓库与文中提到的演示视频未核实。
- **通读：** 已完成

## 开源状态

- 代码 + 权重 + 语义查找表：论文 Sec. I 称在 GitHub 提供；仓库 README 称权重在 Hugging Face（lucashoefs/sam-vtp，未核实），MIT 许可，SAM 2 以子模块方式使用（Apache 2.0）。
- README 的 Citation 一栏为 TBD。

## 核心摘录（均出自论文，带章节）

1. **问题（Sec. I）：** 多数视觉可通行性方法逐帧预测，视角变化 / 遮挡下时序不一致；循环模型把历史压成定长状态，下游融合作用于已预测的图，二者都不让稠密预测器直接访问先前编码的视觉特征。
2. **架构（Sec. III-A，Fig. 2）：** Hiera-T + FPN；记忆库 N=7 帧；学习得到的 4 个稀疏 token 加稠密嵌入替代 SAM2 提示编码器；关闭多 mask 输出、只监督单 token；解码器 stride 4，加残差头在全分辨率上细化；深度分支（5 通道：斜率、局部高度粗糙度、高度、log 距离、有效性）加到最粗层特征。无位姿 / 里程计输入。
3. **监督（Sec. III-C）：** 离线生成 [0,1] 稠密目标 y = 语义先验 × 坡度 / 粗糙度衰减，并对高出局部地形 0.55 m 或超出 (-1.0, 2.5) m 的像素置 0；坡度 12°→35° 线性降到下限 0.3，粗糙度 0.03→0.20 m 降到下限 0.4；语义先验为手工全局查找表（七档，0.00-1.00），论文明说数值是序数的、未校准。目标是「语义 + 几何可通行性」而非物理交互测量。
4. **损失（Sec. III-A4）：** 模糊 Jaccard（Ruzicka）损失 + BCE + Focal(γ=3) + 时序一致项，权重 (1.2, 0.8, 0.7, 0.5) 由验证集粗搜得到。
5. **数据与训练（Sec. III-B, III-D）：** TartanGround，136 条轨迹 / 24 个环境，94 训练 / 20 验证 / 22 测试（按环境分层）；T=24 帧、stride 2，AdamW 基础 lr 3e-5，单卡 RTX 5090。
6. **主结果（Sec. IV, Table I，测试轨迹，SAM 行为 3 个种子均值）：** fJacc 0.908 → 0.924，二值 IoU 0.854 → 0.886，MAE 0.024 → 0.018，ECE 0.013 → 0.010，Flicker 0.0175 → 0.0120（文称降 31%），种子间标准差均 <0.001。WayFASTER / WVN 为单次运行，fJacc 0.791 / 0.760；GeNIE 原权重零样本 0.667。
7. **吞吐（Table I）：** SAM-VTP 296 FPS（RTX 5090）/ 30.1（AGX Orin）/ 21.4（Orin Nano）；论文自行重实现的 SAM-TP 为 540* / 44.9* / 33.5*（带星号为优化过的实现）。
8. **消融（Table II，较短训练预算，1000 个序列窗口）：** fJacc 帧间基线 0.875，+几何 0.900，仅记忆 0.894，两者 0.914；撤去推理时深度 0.853（正文 / 结论称损失 0.061）；二值目标 0.733。Orin Nano 上记忆多 14.9 ms，几何多 2.1 ms。EMA 平滑的 flicker 低于 SAM-VTP 但精度明显下降。
9. **实机闭环（Table III）：** JetAcker（阿克曼转向）+ RealSense D455f + Jetson Orin Nano 8 GB，两种方法共用 GeNIE 的弧线规划器与纯追踪控制器。铺装场地 15 次、粗糙地形 21 次运行。无可行路径周期占比：铺装 32.4% → 9.9%，粗糙 20.6% → 3.0%（摘要写 3.3× / 6.9×）。SAM-TP 仅用 RGB，SAM-VTP 用 RGB-D，故不能分离记忆与深度。运行由操作员终止，不是固定目标；粗糙地形进度率与距离只在 8/10、7/10 对上分离，底盘被草丛架住时间隙而非感知是瓶颈。
10. **论文自述局限（Sec. V）：** 部署需深度传感器；目标绑定一个间隙包络，平台差异大时需重新生成目标；序列初始化有预热；所有分割精度都是合成数据、对照自家管线生成的目标；闭环只测规划行为不测 mask 质量；管线需要逐像素语义图，仿真提供而真实影像没有，用于仿真之外需语义分割器在环。

## 对 wiki 的映射

- [paper-sam-vtp](../../wiki/entities/paper-sam-vtp.md)
