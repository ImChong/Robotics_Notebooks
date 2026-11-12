# Tapestry（arXiv:2610.36823）

> 来源归档（paper）

- **标题：** TAPESTRY: A Single Backbone for Video Generation and Robot Control（缩写展开：Tokenizing Actions as Video Patches in a Shared Transformer Policy）
- **类型：** paper
- **作者（首页）：** Jeremy A. Collins, Seungjae Lee, Rachanon Wachakorn, Krishnan Srinivasan, Animesh Garg, Vitor Campagnolo Guizilini, Paarth Shah（Georgia Tech / University of Maryland / Toyota Research Institute）
- **会场（首页）：** 10th Conference on Robot Learning (CoRL 2026)
- **arXiv：** <https://arxiv.org/abs/2610.36823>（v1，2026-10-27）
- **PDF：** <https://arxiv.org/pdf/2610.36823>
- **项目页：** <https://tapestry-wam.github.io/>
- **入库日期：** 2026-10-28（2026-11-12 按原文通读升级）
- **线索来源：** [人类视频怎样变成机器人能用的动作经验？](../blogs/wechat_embodied_station_59_papers_human_video_to_robot_experience_2026-10-28.md)（具身智能小站，2026-10-28）；Glen Berseth CoRL 2026 阅读清单（P2，World Models & World Action Models 组）
- **一句话说明：** 把每个动作时间步当作视频 VAE 潜空间中的一个单补丁标记，让预训练视频扩散 Transformer 直接去噪动作，部署时只走动作通路并单步 DDIM。
- **核对范围：** 下载 arXiv v1 PDF（22 页，首页标题与编号核对一致），用 pdftotext 逐段读完正文 §1–§6、全部附录 A–G（含 G.1–G.10）、Table 1–10 与图注；图 1–8 只读图注与图中文字，未逐图细看像素细节；参考文献只读条目、未逐篇核对被引内容；未运行任何代码，项目页未重新抓取（项目页状态沿用 2026-10-28 的核查记录）。
- **通读：** 已完成

## 开源状态

- 论文正文只给出项目页 tapestry-wam.github.io，未给 GitHub 仓库地址。
- **项目页未列 GitHub**（步骤 2.5 核查，2026-10-28）：项目页 HTTP 200，未见 GitHub 链接或 coming soon 字样。

## 核心摘录（均来自原文，括号为章节 / 表号）

1. **动作令牌化（§2.1）：** 每个动作向量 a_i 经线性投影 W、无仿射 LayerNorm、骨干共享的补丁卷积 φ，得到一个 DiT token；读出时用 W 的转置与 φ_out。W 仅 2·2·16·20 = 1280 个参数（无 bias），是动作通路相对预训练骨干唯一新增的模块。直接线性投影到 DiT 隐维度早期训练不稳定；去掉 LayerNorm 在 5 epoch 消融中降到 33%（§4.4，Table 6）。
2. **骨干与数据（§2.4，附录 D/E）：** CogVideoX-2B，冻结 3D-VAE 与 T5-XXL；32 张 H200，75k 预训练步加 25k 微调步，有效 batch 128。视频源：DROID、BridgeData v2、AgiBot-World、Ego4D、Epic Kitchens、SSv2、Panda-70M，全部只走视频目标；唯一动作标签来自域内双臂遥操作（约 6.02 小时，占每个 batch 权重 0.5，Table 8）。作者称把跨本体机器人数据的动作标签也路由进来反而让域内动作表现更差（附录 E）。
3. **三目标训练（§2.2，式 (1)）：** 视频扩散、联合视频加动作扩散、纯动作扩散；(λv, λa) = (0.1, 1)；有动作标签的样本以 p_joint = 0.5 走联合模式，否则同时走视频与纯动作模式。联合模式用共享扩散时间步与非对称注意力（动作可看视频，视频不可看动作）。
4. **单步推理（§2.3，§3，附录 G）：** 零终端 SNR 加 v-prediction 下，从 t = T 的一步 DDIM 化简为 x̂0 = −vθ(ε, T, τ, xo)（式 (2)），输出近似条件均值。作者观察到多步 DDIM 在动作块上引入高频抖动，并归因于视频预训练的频率先验与动作不符（自然视频 f95 约为动作块的 4–6 倍，Fig.2a，Table 10）；已排除 AdaLN 耦合、解耦时间步、调度器与 v-prediction 参数化四种替代解释（G.4）。
5. **LIBERO 学习自视频协议（§4.1，Table 1）：** 仅 LIBERO-90 有动作标签，LIBERO-10/Object/Spatial/Goal 只有视频；每套件 100 次 rollout。TAPESTRY 平均 76%（55/95/87/65），Cosmos Policy 70、DreamZero 70、AMPLIFY 61、从头训练变体 58；LIBERO-10 上 Cosmos Policy 70 与 DreamZero 60 均高于 TAPESTRY 的 55，Goal 上 DreamZero 66 略高于 65。作者说该协议下绝对数字与标准全监督 LIBERO 榜单不可比。
6. **真机（§4.2，Table 2/3，附录 B）：** 20 自由度双臂 I2RT YAM 平台，腕部与第三视角相机。域内三任务（Toolbox、Oven、Corn）各 10 次 rollout：TAPESTRY 28/30，Diffusion Policy 15/30，去掉大规模视频协同训练 20/30，去掉视频生成目标 19/30。三种 OOD 条件各 5 次 rollout：TAPESTRY 完整成功 10/15，DreamZero 4/15（只做成倒球），其余 0/15；作者自注与 DreamZero 之差 p = 0.07 不显著。
7. **推理开销（§4.3）：** 单张 RTX 5090，16 步动作块 84 ms、约 9 GB 显存；DreamZero（5B）发布代码约 20 s 每块，加文本嵌入缓存与激活检查点后 3.7 s，作者称约 44 倍慢；DreamZero 要 4 次前向且骨干更大（5B 对 2B）。训练算力：真机设置 TAPESTRY 1.9k H200 小时、DreamZero 2.4k；LIBERO 设置 TAPESTRY 2.5k、DreamZero 1.1k、Cosmos Policy 7.1k（附录 D）。
8. **作者自述局限（§6，G.7–G.8）：** 单步推理返回条件均值，每个观测一个确定性预测，多模态或高随机动作任务可能需要迭代细化；结果依赖骨干的零终端 SNR v-prediction 调度，在 Wan 的整流流骨干上替换原目标不稳定，单步是否成立未解；W 按 20 维双臂动作空间定尺寸，按本体分别设动作头未探索。

## 对 wiki 的映射

- [paper-tapestry](../../wiki/entities/paper-tapestry.md)
- [潜在动作、免渲染事件仿真与机器人评测：具身智能小站 10-28 期 59 篇阅读坐标](../../wiki/overview/embodied-station-2026-10-28-59-papers-technology-map.md)
