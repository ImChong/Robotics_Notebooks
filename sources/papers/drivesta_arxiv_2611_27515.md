# DriveSTA：Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving

- **类型：** paper / end-to-end autonomous driving
- **中文标题：** 超越单条人类示范：从带评分的备选轨迹学习端到端驾驶
- **正式标题：** Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving
- **作者：** Xinyao Rao、Zecheng Tang、Liuxuan Sun、Zheyuan Lai、Minjun Yu、Xun Cao、Wei Li（通讯作者）
- **机构：** 南京大学、北京大学、EABOT.AI
- **论文：** https://arxiv.org/abs/2611.27515
- **HTML（原文）：** https://arxiv.org/html/2611.27515v1
- **alphaXiv（同一预印本的阅读入口）：** https://www.alphaxiv.org/abs/2611.27515
- **PDF（原文）：** https://arxiv.org/pdf/2611.27515v1
- **版本：** arXiv:2611.27515v1，cs.CV，2026-11-20；预印本，非独立同行评审结论
- **项目主页：** https://rtwotwo.github.io/DriveSTA/
- **代码：** https://github.com/Rtwotwo/DriveSTA
- **模型/数据入口：** https://huggingface.co/Rtwotwo/DriveSTA（[发布清单与模型卡核对](../models/drivesta-release-assets.md)，未下载权重或复现实验）
- **入库日期：** 2026-11-23
- **核对范围：** 已读取用户提供的同篇 21 页 PDF（正文第 1–11 页、Supplementary Material 第 12–21 页），逐节核对 Introduction、Related Work、Method §3.1–3.5、Experiments §4.1–4.4、Discussion/Limitations、Conclusion、Appendix A 数据与指标协议、Appendix B 搜索/训练/JEPA/消融/时延、Appendix C 可视化及 Table 1–8、S1–S4。与既有官方项目页和官方仓库静态审读记录交叉校验；未独立执行训练或 benchmark。
- **通读：** 已完成

## 核心问题与贡献（正文 §1、§3，图 1–2）

每个场景只有一条实际驾驶轨迹，不意味着其它行驶决策错误。DriveSTA 继承 ChainFlow-VLA 的自回归候选生成与残差流精修，但用**离线仿真评分的多样化轨迹**同时提供：①互异 proposal 的几何训练目标，②局部残差精修目标，③含成功/失败样本的候选评分标签。另以**共享且适配过的 JEPA 历史视频编码**取代在线大 VLM 条件推理。

输入为四个当前相机视角、前视相机的已观察帧、自车历史状态和导航指令；输出选定轨迹为 **8×(x,y,heading)**，以 0.5 s 为间隔覆盖未来 4 s。候选数 **64**，Stage 2 对全部候选执行 **12 步 DDIM** 后重评分再选最优（正文 §3.1、§3.5）。

## 离线备选轨迹构造（正文 §3.2，附录 B.1）

- **连续变形：** 以人类示范前端为锚，将终点纵横向偏移 `u` 沿 8 个时刻逐步施加，平滑步进函数 `b(r)=6r^5-15r^4+10r^3`，保留起始位置且两端位移导数为零（式 1）。采样时刻固定，因此变形同时影响速度；航向随轨迹切向更新。
- **搜索与评价：** 每批 64 次查询中 25% 广域探索，其余聚焦高分邻域；预算由 **3,072 → 8,192 → 32,768** 次递增。每场景保留跨分数区间的 3,072 条记录用于评分训练，准入需至少 256 条分数 ≥95 的路径。
- **几何筛选：** 分项 NC/DAC/TTC/Comfort/DDC 全分，再检查方向、车辆轮廓、运动连续性、跟踪误差；要求轨迹总分 `J≥0.95`、相互全轨迹 ADE 至少 **0.25 m**；上限 **32** 个目标，最多 256 次跟踪校验。**7,265** 个没有合格筛选目标的场景仍保留专家与评分监督。
- **规模：** 103,263 个训练场景；其中 **95,998** 个含筛选目标，共 **2.61M** 条。注意附录 B.6 的 **317.3M** 是搜索归档总路径数，不是 2.61M 筛选训练目标；另 **55.45%** 的含目标场景存在评分高于人类参考的备选路径。

## Stage 1：多目标 Proposal + 评分（正文 §3.3）

- **感知与生成：** DINOv2 + LoRA 编码四路当前视角；自回归模型通过 GRU 与自行车模型积分提出 64 条轨迹。
- **多解匹配：** Hungarian 一对一地把轮换的最多八条高质轨迹匹配到不同 proposal（式 2）；几何距离是平均 `|Δx|+|Δy|+0.25|wrap(Δθ)|`，`L_G=L_E+0.25 min(1,e/2)L_A`。相对于只监督最近示范，这直接改善候选覆盖。
- **质量打分：** 评分头预测 **NC、DAC、TTC、Comfort、DDC、EP** 六分项；NAVSIM v1 的预测聚合为 `NC × DAC × (5EP+5TTC+2Comfort)/12`，DDC 参与监督但不进入 v1 聚合（式 3、附录 A.2/S1）。
- **负例与排序：** 对原生 64 候选重新仿真评分，同时混入 64 条离线高/中/低/零分回放。可选 pairwise ranking 仅在混合数据主配置启用；真实数据主配置不加该项，不能笼统写成“排序损失必然有益”（正文 §3.5、表 4）。

## Stage 2：多目标残差精修 + 共享 JEPA（正文 §3.4–3.5）

- **兼容性门控：** 每个 proposal 最多选四条高质量备选作为残差目标，要求位置 ADE ≤3 m、终点 ≤6 m、平均航向误差 ≤0.5 rad；残差是备选轨迹减 proposal（航向取 wrap）。
- **去噪目标：** 在候选兼容目标中按**重建误差最小**选定一个，使用 DiT 的 `v-prediction` 与几何目标共同监督（式 4–5）；无法匹配者以弱权重 0.05 学习**零残差**，降低误修风险。
- **历史时序：** JEPA 自 Drive-JEPA 初始化，以 **8 帧、2 Hz** 的 masked video-feature prediction 适配；回归 EMA 教师的归一化特征。规划模块只使用“此前+当前”的**已观察**前视帧，且 JEPA 编码只算一次供 DiT 与 scorer 共享；训练未来帧仅参与 JEPA 表征学习，不泄露给在线规划（式 6、附录 B.3）。
- **重评分：** 对 64 条 proposal 实际执行 target-free DDIM 精修后**重新取得仿真标签**训练评分器。训练中用到的 archive、真实分数、未来帧、JEPA predictor、EMA encoder 均不进入在线推理（正文 §3.5）。
- **模型设定：** DINOv2 ViT-S/14（LoRA rank 32）；JEPA ViT-L（宽 1024）；DiT 12 blocks×8 heads，训练使用八张 H20 GPU；附录 B.2、Table S2–S3。此为作者训练报告，不表示本仓库已复现。

## 实验与消融核对（正文 §4，表 1–8；附录表 S4）

| 版本 | NAVSIM v1 Navtest PDMS | HUGSIM RC | HUGSIM HD-Score |
|---|---:|---:|---:|
| DriveSTA S1 | 94.76 | 51.9 | 36.9 |
| DriveSTA S2 | 95.14 | 55.2 | 44.0 |
| DriveSTA-Scale S1 | 95.09 | 54.0 | 44.0 |
| DriveSTA-Scale S2 | **95.55** | **56.1** | **46.7** |

- **数据口径：** Base 103,263 NAVSIM 训练场景；Scale 另加 129,409 SimScale，合计 **232,672**；Navtest 共 **12,146** 场景。作者在表 1 所列**发布基线**中，Scale S2 的 PDMS 四舍五入为 95.6（领先该表）；官方项目页展示的另一组同类对照含 **FPD-Drive-Pro 95.69**，所以不得跨不同实验/精度口径称“所有模型绝对第一”。
- **闭环：** HUGSIM 是 **345** 条匹配 episode；四个难度组 E/M/H/X 为 49/126/86/84。Scale S2 的平均 RC 56.1、HD-Score 46.7 在表 2 的对照中领先，但 **Extreme 组**从 S1 的 30.3 降到 S2 的 **27.3** HD-Score，不能夸称所有场景都改善。
- **消融：** 表 4 显示 Dense 在真实数据上提高候选 Oracle +0.49、所选 PDMS +0.64；Replay、Ranking 在真实数据并非单调提升。表 5 显示多目标局部筛选、共享视频条件及 JEPA 适配分别产生增益；表 6 显示视频观测恢复可使固定候选池的 S2 评分提高 +3.20 PDMS。
- **NAVSIM v2：** 仅用 **S1** 评测；Navtest EPDMS Base/Scale 为 **91.15/91.32**，低于 WA-JEPA 的 **91.7**；Navhard EPDMS Base/Scale 为 **50.80/56.13**，后者低于 DriveZero-Scale 的 **57.1**。v2 额外惩罚跨次规划不舒适，不能把 v1 PDMS 排名直接移植到 v2。

### 73% 推理时延：已核实（正文第 6–7 页 Table 3，附录 B.7）

| 方法 | 平均前向时延（ms） | P95（ms） | 吞吐等效（Hz） |
|---|---:|---:|---:|
| ChainFlow-VLA | 1294.68 | 1302.79 | 0.77 |
| DriveSTA S1 | 148.99 | 150.32 | 6.71 |
| DriveSTA-Scale S2 | 349.30 | 359.29 | 2.86 |

**相对 ChainFlow-VLA 下降约 73%**，计算为 `(1294.68−349.30)/1294.68≈73.0%`。这是作者的 **H20、batch 1、同样 64 候选及 12 DDIM 步、12,146 Navtest 场景**的实现级前向比较，**只**包括 GPU resident 输入到选定轨迹（包含编码、生成、精修、评分与选择），不包含磁盘、预处理、传输和仿真执行；ChainFlow-VLA 使用 BF16 InternVL，其 VLM 与注意力实现及数据流有差异，故不能将该比例简单解读为模型架构单变量加速或车端控制频率保证。

## 局限和实践边界（正文 §5，附录 A/B）

1. 搜索由人类参考轨迹的平滑形变定义，覆盖不了所有机动行为；评分与有效性依赖 evaluator，非真实道路安全验证。
2. 离线对“已记录周围交通”的模拟不能真实表达对抗/交互反应；HUGSIM 仅做特定闭环模拟。
3. 候选逐条训练不直接优化相邻 replanning 的一致性；v2 Expanded 训练的 Extended Comfort 下降，表明单轨迹分数与舒适性不等价。
4. 报告采用每配置**一个训练随机种子**，比较基线主要使用公开论文分数而非全部重新共同复现，注意实验不确定性。
5. 仅静态核对源码 CLI / 模块；未在本次环境执行 NAVSIM/HUGSIM，也未据此宣称具备实车部署认证。

## 对 wiki 的映射

- [DriveSTA 独立论文实体](../../wiki/entities/paper-drivesta.md)
- [官方项目主页归档](../sites/drivesta.md)
- [官方代码仓库归档](../repos/drivesta.md)
