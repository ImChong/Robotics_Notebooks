# RARM 论文来源归档

## 元数据

- **题目：** RARM: Confidence-Gated Progress Reward Modeling for RL in Manipulation
- **作者：** Pengzhi Yang, Xinyu Wang, Pengyu Jing, Kehan Wen, Yiduo Qu, Zhenhao Huang, Minghao Fu, Xin Liu, Yaheng Shen, Fan Shi（首页标注：多数作者署 NUS Human-Centered Robotic Lab；Xinyu Wang 署 Booking.com；Yiduo Qu 同署剑桥大学；Minghao Fu 署南京大学人工智能学院；前五位标共同一作，Fan Shi 为通讯作者）
- **预印本：** https://arxiv.org/abs/2606.22027（cs.RO，v1，2026-06-20）
- **PDF：** https://arxiv.org/pdf/2606.22027v1
- **项目页：** https://rarm-robotics.github.io/（页面自称 Accepted at CoRL 2026；列出 GitHub 与 Hugging Face 链接）
- **代码：** https://github.com/pengzhi1998/rarm（main 分支含奖励模型训练与片段相似度推理，README 说明 DrQ-v2 策略训练在 `policy_training_RARM` 分支；本次只读了 main 分支 README，未读代码，未核对 HF 权重内容）
- **入库日期：** 2026-11-12
- **一句话说明：** 用一条成功示范当「进度锚」，由一个在通用视频上自监督训练的片段比较器定位 rollout 在示范中的位置，只对置信度过阈值的前向进度发奖励的稠密奖励模型。
- **核对范围：** arXiv v1 PDF 全文：摘要、§1 引言、§2 相关工作、§3.1-3.3 方法、§4.1-4.3 实验、§5 结论、§6 局限、附录 A（仿真任务表）、B（真机硬件与四个任务）、C（超参数、训练数据、DrQ-v2 网络）、D.1 合成参考、D.2 视觉扰动、D.3 推理开销、E（DrQ-v2、π0、DSRL 细节）。图（Figure 1-11）只读到图注和正文对图的描述，未逐图核对曲线，数值以正文、Table 1 与 Table 6 为准。附录 A 的任务缩略图未看。没有读：参考文献逐条内容、项目页的视频与交互内容、代码仓库源码、HF 权重。
- **通读：** 已完成

## 论文要解决的问题

机器人操作 RL 的瓶颈是奖励设计：稀疏成功奖励监督弱，手工稠密奖励难写且难泛化（摘要、§1）。基于进度的学习式奖励模型有前景，但已有方法常需任务专属示范或进度标签，并可能给「视觉上像、物理上错」的状态高奖励，被 RL 智能体利用（摘要、§1）。

## 方法要点（作者自述）

- **训练（§3.1，附录 C.2）：** 片段长度 T=5 帧；从无标注视频取锚片段，正样本取自同一锚区域，负样本三类：同视频不相交区域、锚区域的时间反转片段、其他视频的片段。损失为软最近邻损失（式 1），温度 τ=0.1。增强为随机缩放裁剪和颜色抖动（片段内一致），同视频片段共享水平翻转。训练语料取自 DAVIS 与 MOSE 两个视频目标分割数据集，滤掉平均相邻帧掩码 IoU 超过阈值的近静态序列（附录 C.2）。
- **结构（§3.2）：** 每帧用带 LoRA 的 DINOv3 编码，patch 特征空间池化，加时间旋转位置编码，经一层帧间自注意力得到片段表示（孪生共享）；比较器用交叉注意力（X 作 query，Y 作 key/value）并投影到低维，轻量头输出标量 g。比较有方向性：「rollout 片段相对参考片段如何」。
- **奖励（§3.3）：** 参考示范上滑窗得到 M 个重叠片段，第 j 个按其归一化时间位置赋进度 π_j∈[0,1]。当前 rollout 片段与全部参考片段打分，取 argmax 得 j*（式 2，不要求时间戳同步）。每个参考片段有离线标定的置信阈值 δ_j（来自参考示范自身的匹配分数统计，图 2 说明与 §3.3）。若 g≥δ_j* 则进度 p_t=π_j*，否则 p_t=p_{t-1}；奖励 r=1{p_t>p_{t-1}}（式 3）。低置信匹配与重复访问同一进度都得零奖励。
- **RL 接入（§3.3、附录 E）：** 仿真从零训练用 DrQ-v2；真机用 DSRL 在冻结的 π0 上学习注入噪声（SAC 实现），RARM 替换 DSRL 默认的 −1 奖励。

## 实验设置（§4、附录 B、C、E）

- **仿真（§4.1）：** 4 个 MetaWorld 任务与 5 个 LIBERO-10 任务（共 9 个；图 3 子图为 MetaWorld 的 coffee pull、button press wall、drawer open、soccer，LIBERO-10 的 book in caddy、stove + pot、bowl in drawer + close、mug + pudding、mug in microwave + close）；DrQ-v2，1M 环境步（正文印成「106」，应为 10^6 的排版丢失），5 个随机种子，每 10k 步评测，每种子 10 条 rollout；各方法使用「自己的稠密奖励加任务成功信号」。
- **基线（§4.1）：** 参考类 RoboCLIP、TemporalOT；VLM 类 GVL、RoboDopamine、RoboMeter（Robometer）；消融「Similarity Reward」（直接用最大相似度当步奖励）。图 5 与附录 D.3 另含 VIP。
- **真机（§4.3、附录 B）：** AIRBOT Play 双臂；π0 + DSRL；任务 Pick Eraser、Drawer Open（单臂短程），Bimanual Hand-over、Bimanual Clothes Folding（双臂长程，后者三阶段）；训练预算 30/60/90 条 on-robot rollout，每个检查点评测 10 条 rollout。RARM 比较器输入 384×384，π0 输入 128×128，奖励约 5 Hz（步长 4）。

## 报告的结果（论文自报）

- **仿真（§4.1、图 3）：** 正文称 RARM 在两个基准上整体最强；短程 MetaWorld 上若干基线最终也能学会，但 RARM 更快更稳；LIBERO-10 长程上 RoboCLIP 与 TemporalOT 表现差。结果以学习曲线给出，正文没有汇总数值表。
- **消融（§4.1、图 4）：** 在一个 LIBERO 任务与一个 MetaWorld 任务上做四个组件消融（置信门、交叉注意力、时间处理器，门过严也有损失），所有变体都弱于完整模型。
- **失败/成功配对 rollout（§4.2、图 5，叠衣任务）：** 以各模型自身成功 rollout 的最终累计奖励归一化，RARM 在失败 rollout 上累计到成功的 0.50；GVL 0.97、Robometer 0.88、RoboDopamine 0.84、VIP 0.82；去掉阈值的变体在失败 rollout 上累计约为成功的 1.89 倍。只展示了一对 rollout。
- **真机（§4.3、Table 1，每格 10 条评测 rollout）：** 基础 π0：Drawer 2/10、Pick eraser 2/10、Hand over 2/10、Clothes folding 1/10。DSRL 稀疏奖励（BinR）对 RARM，在 30/60/90 条 rollout 时：Drawer open 4/6、6/7、8/10；Pick eraser 3/6、5/8、8/9；Hand over 2/5、4/6、5/6；Clothes folding 0/1、1/3、2/8（格式 BinR / Ours，数值取自 Table 1 的文本抽取，已与正文引用的 6/4、6/3、5/2、2/8 等数字交叉核对）。
- **推理开销（附录 D.3、Table 6）：** 125 帧一条轨迹上每帧耗时 RARM <10 ms、5100 MB 显存；GVL <50 ms / 13200 MB；Robometer <100 ms / 13200 MB；RoboDopamine <300 ms / 19000 MB；VIP、RoboCLIP、TemporalOT 均 <10 ms，显存 4200 / 5400 / 6600 MB。
- **参考视频鲁棒性（附录 D.1、图 10）：** 仅在 MetaWorld pull 一个任务上，参考换成不同物体的真实视频（2 种物体各 2 种子）或由 SeeDance 2.0 据首帧图和文字指令生成的视频（4 种子），作者称渐近成功率与样本效率与真人参考相当。
- **视觉扰动（附录 D.2、图 11）：** 把一条叠衣成功 rollout 数字化加眩光、绿色滤镜、黑块遮挡、视角扭曲，用另一条未改动参考估计进度，作者称估计进度贴近理想线性趋势。定性图，无量化指标。

## 作者承认的局限（§6）

大幅度的本体、视角或场景变化可能使纯视觉比较退化；细粒度接触、抓取稳定性、力交互、细微形变难以从 RGB 判断；基于参考轨迹的进度使重复/循环任务困难，需多参考或显式记忆（未来工作）。结论 §5 另承认假设能拿到有代表性的参考示范。

## 阅读时发现的文内不一致（非论文自述）

- 附录 B 的真机回合超时为 Hand-over 600 步、Clothes folding 1200 步，附录 E.3 表中 max timesteps 为 300 与 1000；两处口径未说明如何对应（E.3 另列 query steps，可能是不同计量）。
- 图 3 把消融叫 AblationSim，正文与图 5 说法为 Similarity Reward；图 5/表 6 含 VIP，但 §4.1 基线列表没有 VIP。
- 图 4 图注称「four component ablations」，正文只点名三个组件（置信门、交叉注意力、时间处理器）并另提门过严的情形，四个变体的对应关系需看图确认，本次未核。

## 对 wiki 的映射

- [paper-rarm-confidence-gated-progress-reward](../../wiki/entities/paper-rarm-confidence-gated-progress-reward.md)
