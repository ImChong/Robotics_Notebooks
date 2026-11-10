# EgoWild2Dex：从野外人类经验学习灵巧机器人操作 / Learning Dexterous Robotic Manipulation from In-the-Wild Human Experience

- **类型：** paper / 原文阅读归档
- **论文：** https://arxiv.org/abs/2609.23755
- **原文版本：** arXiv:2609.23755v1，2026-09-20
- **全文 HTML：** https://arxiv.org/html/2609.23755v1
- **PDF：** https://arxiv.org/pdf/2609.23755
- **官方主页：** https://mmlab.hk/egowild2dex/
- **机构：** 香港大学（The University of Hong Kong）；Kinetix AI
- **作者：** Kunyang Lin、Xutao Wen、Jingxi Lin、Lanyong Lin、Jiaming Liu、Tianshuo Yang、Xianchi Chen、Yue Han、Yiduo Li、Zhanpeng Zhang、Ping Luo
- **通读：** 已完成
- **核对范围：** arXiv v1 HTML §I–VI（引言、相关工作、方法、实验、结论、局限），Algorithm 1，Fig. 1–5、Table I–IV；附录 A（硬件、动作对齐、GeoFormer、训练、执行及 DAgger），附录 B（基线、Table IX–X、动作分布/DTW 对齐），附录 C（EgoWild 质量评估/筛选、Fig. 8–10），附录 D（作者贡献），并与作者项目主页各板块交叉核对。因 PDF 文件过大，逐段阅读采用 arXiv 官方 HTML 全文，不声称对 PDF 版面逐页核查。
- **复核日期：** 2026-11-10

## 研究问题与方法边界

- **两项对齐：** 头戴相机的动态 ego 视角 vs 固定/腕部 robot 相机视角；人的腕指运动 vs 双臂灵巧手的关节与电机指令。
- **EgoWild：** 经过筛选的 538.9 小时、179,049 episodes、125,961 唯一任务描述、1,282 个物体类、约 58.2M 帧（30 Hz）。原始高分辨率视频为 2048×1536，含双手追踪及三个时间尺度的语言标注。家庭、超市、包裹站、工厂、药店、办公室等真实场地，不预先整理布景。
- **不可混淆：** 这不是“纯 RGB 视频直接无监督变成机器人动作”；数据采集用 PICO 手部追踪和每手一个校准辅助 tracker，且任务迁移还需要 task-specific ego、glove、robot 专家与 DAgger 数据。

## §III-A GeoFormer（Eq. 1–5，Fig. 2）

1. 同一目标任务的 ego 与 robot 图像池随机配对，**无需逐帧对齐**，但必须任务相关。八维参数 p 构造 traceless 3×3 矩阵 P(p)，指数映射得到单应矩阵 T=exp(P(p))。
2. 五级独立的 CNN+MLP 预测器，输入 ego 图和当前 warp 后的 robot 图，逐次预测八维残差，更新 p。训练方向为 **robot → ego**，将 robot 内容按 T warp 后与 ego 背景合成，Wasserstein critic 判别几何自然程度；损失额外约束残差大小和有效区域覆盖。
3. 训练完成丢弃 critic、冻结 GeoFormer；策略联合训练用 **T 的逆矩阵** 将 task-specific ego 观察对齐机器人视角。它是二维单应几何校正，不是显式重建 3D 场景，更不能重建头动造成的画面外缺失物体。
4. 附录 A.3 参数：GeoFormer 输入 240×320；N=5；矩阵指数近似保留 20 项；5000 iteration、batch 8、predictor/critic LR 均 1e-6，覆盖率目标 [0.4,0.8]。

## §III-B–D 三阶段 / 统一动作 / 闭环

- **动作标签 Eq. 6–8：** A_t=[left arm targets; left hand motor commands; right arm targets; right hand motor commands]，维度 2(d_arm+d_hand)。标定初始参考姿态，腕位姿经相对变换与约束 IK 成为臂关节命令，手指通道 retarget 为手电机命令。主实验 2×(6 臂关节+6 手通道)=24 维。学习的是“命令动作”，不是把接触后受阻的实际关节状态误当成动作意图。
- **Stage 1 Human-to-Robot：** 原始 EgoWild 大规模 human-to-robot 预训练视觉语言先验；图 1 标注更新 VLM。
- **Stage 2 Human–Robot co-training：** task-specific ego（GeoFormer 对齐）+ robot-view glove + robot demo，联合更新 VLM 与 flow-matching action expert；每任务 1,000 ego、1,000 glove、100 专家 robot，batch 来源约 1:1:1。
- **Stage 3 Robot refinement：** 使用 robot 专家轨迹进一步训练，并用访问状态采集的 10 段 DAgger recovery 补足实际接触和失败恢复；三阶段共享 image/instruction/joint-state → H=50 action chunk 的条件 flow-matching 目标。
- **执行 Eq. 18–22：** Piper Mixed Integrated Torque Control / 柔顺 PD 力矩执行目标关节指令；相邻动作块 overlap 使用线性权重 blend；接管时 Anchored Delta-Cmd 以 t_take 时刻的**机器人当前指令**为锚点，只累加操作者接管后的相对运动，防止绝对重定向瞬间跳变。含手柄 pivot 标定、滤波/限速、丢帧/IK 异常停机或保持最后安全命令。

### 附录 A.4 训练参数

| 阶段 | 步数 | 全局 batch | peak LR | A100 |
|---|---:|---:|---:|---:|
| 1 Human-to-Robot | 100,000 | 512 | 7e-5 | 32 |
| 2 Co-training | 50,000 | 128 | 3.5e-5 | 8 |
| 3 Robot refinement | 50,000 | 128 | 2.5e-5 | 8 |

均为带 warmup 的 cosine schedule；H=50，Stage 2 三类来源 1:1:1。**机器人示范量小不等于训练计算量小。**

## §IV 实验 / 表格逐项核对

- **平台：** 两条 6-DoF AgileX Piper + 两只 6 通道 BrainCo Revo2 灵巧手，固定 top-down 和双腕 RealSense D435，共三相机，机器人控制 30 Hz。硬件和运行频率见附录 A.1。
- **Table I：** Open-Box 90%（9/10）、Glue-Figure 100%（10/10）、Ice-Water 100%（10/10）；总成功 29/30=96.7%。10 分制任务完成分 9.5/10/10。**不是数百次实验的成功率。**
- **Table II（只在 Open-Box 做消融）：** 仅 robot refinement 0%；加 Stage 1 20%；仅 Stage 2 + refinement 40%；Stage 1+2+3（不带 DAgger）70%；附加 DAgger 90%。不是把 DAgger 增益外推到全部三任务。
- **Table III：** GeoFormer 相对 MoGe+LaMa Project+Inpaint，平均图像对齐处理延迟 **162.58→7.43 ms（21.9×）**；GPU-h/1M **45.16→2.06**；Open-Box 成功 **80%→90%**。21.9×仅限图像处理，不是端到端推理/训练加速。
- **Table IV：** 使用开箱任务变更包裹内容物，3 个已见物体经每物体 <100 新示范适应后平均 70%；3 个未见物体完全不做特定适应，平均 33.3%。跨本体 Marvin Pro（两臂总计多 2 DoF、相同手）任务 I 60%。
- **Table IX：** 任务专用人类 ego 时长依次 5.81/4.16/3.09 h；glove 5.11/4.37/3.07 h；**每任务 110 条 robot（100 expert + 10 recovery）**共 0.64/0.48/0.51 h。论文的「<1 h robot data」**不包含**通用 538.9 h corpus、每任务额外人类数据、对齐标定与训练算力。
- **Table X：** Stage 2 人类 ego、glove、robot 任去一组，Task I 结果均下降。训练中 glove 帮助稳定人机共训优化。
- **附录 C：** PICO 4 Ultra + 辅助手部 tracker，指尖位置中位误差由 4.21 降为 0.66 cm；记录级/片段级质量筛选、对齐后保留；自动标注在 50 个视频的人工抽样中核验。数据是**质量筛选后的操作子片段**而非任意整段走动视频。

## 局限与开放情况

- §VI：仍需真机微调；二维单应无法恢复头部旋转导致的视野外信息；未验证高自由度（如 22-DoF）灵巧手。
- **开放核验（2026-11-10）：** 官方项目页公开论文、视频和结果，未列可下载的 EgoWild 数据集、模型权重或 EgoWild2Dex/GeoFormer 可运行代码入口；论文末尾说 data/models/code will be released。勿把 XRoboToolkit/Piper 第三方依赖当成本工作开源实现，**源码运行时序图不适用**。

## 对 wiki 的映射

- [EgoWild2Dex 唯一实体页](../../wiki/entities/paper-egowild2dex.md)
- [EgoDex](../../wiki/entities/paper-notebook-egodex-learning-dexterous-manipulation-from-larg.md)
- [Motion Retargeting](../../wiki/concepts/motion-retargeting.md)
- [VLA](../../wiki/methods/vla.md)
