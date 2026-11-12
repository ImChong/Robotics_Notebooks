# FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space

> 来源归档（paper，全文通读）

- **标题：** FlowDAgger: Human-in-the-Loop Adaptation of Generative Robot Policies in Latent Space
- **类型：** paper
- **作者：** Michael Murray, Daphne Chen, Simran Bagaria, Dean Fortier, Tess Hellebrekers, Galen Mullins, Harshavardhan Gajarla, Oier Mees, Maya Cakmak, Andrey Kolobov（Microsoft Research / Microsoft / ETH Zurich / University of Washington）
- **arXiv：** <https://arxiv.org/abs/2607.08877>（v1，2026-07-09，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2607.08877>
- **代码：** https://github.com/microsoft/FlowDAgger
- **项目页：** <https://microsoft.github.io/FlowDAgger>
- **入库日期：** 2026-11-12
- **核对范围：** arXiv v1 PDF 全文：摘要、§1 引言、§2 相关工作、§3 背景、§4 方法（4.1–4.3）、§5 实验（5.1–5.5，Table 1–4、Fig. 3–4）、§6 局限、§7 结论、附录 A（A.1–A.3）与附录 B（B.1–B.4，Table 5–6、Fig. 5–6）；参考文献列表仅浏览。图只能从 pdftotext 的文字与图注读到，未逐像素看曲线；未运行代码，未读 GitHub 仓库与项目页。
- **通读：** 已完成
- **一句话说明：** 冻结生成式策略（流匹配 / 扩散，含世界动作模型），把人类干预动作经“动作反演”还原为基座策略的噪声目标，再用监督回归训练一个小的噪声策略去选噪声。

## 核心摘录

1. **问题（§1）：** 预训练生成式策略在具体部署中会因陌生物体、新动力学、本体差异与长尾情形失败；补采演示再微调成本高且会侵蚀已有技能；动作空间 DAgger 要改整个生成策略；残差法在无约束动作空间加修正，可能离开基座策略可靠行为；潜空间 RL（DSRL）样本效率高但依赖奖励与自主探索。
2. **动作反演（§4.1）：** 基座 `π_gp(s, w)` 用 K 步 Euler 积分（式 2）从噪声 w 得动作块 a。对每步隐式方程 `x_k = x_{k+1} − Δt·v_θ(x_k, t_k, s)` 做不动点迭代（式 3），从 `x_K = a*` 逆推到 `x_0 = w*`。当 `Δt·L < 1`（L 为 v_θ 的 Lipschitz 常数）时迭代是压缩映射，默认 M=5（附录 B.2）。每条修正代价 K·M 次 v_θ 评估，无需对基座反向传播。论文强调少步动作头（K=10，Δt=0.1）下单步显式反推的 O(Δt) 误差不可忽略。
3. **世界动作模型（§4.2，附录 A.1）：** 以 Cosmos-Policy 为例，生成变量是联合潜张量（16×9×28×28），动作、未来状态、价值估计是其中的帧。反演目标为基座预测的干净潜变量 `x0_base`，仅在动作帧加上“专家动作减基座动作”的差值，状态与价值帧保持基座输出。Cosmos-Policy 用 EDM 式调度（σmax=80，σmin=4）：逐步反演同样用不动点，终端去噪 D(·, σmin) 用短 Adam 求解，这是整个反演中唯一对基座模型反向传播的部分。
4. **噪声策略与训练（§4.3）：** 噪声策略 π_w＝观测编码器＋MLP，部署时确定性，动作 `a = π_gp(s, π_w(s))`（式 4），损失 `‖π_w(s) − w*‖²`（式 5）。数据集 D 分两个缓冲区：干预缓冲区（反演得到的 (s, w*)）与自主缓冲区（成功自主 rollout 各转移所用的噪声），每个训练批两者等比例采样。
5. **仿真设置（§5.1）：** MetaWorld；基座 π0.5（动作头 VLA）与 Cosmos-Policy（WAM）；基线含冻结基座、SFT（全权重 BC，离线演示）、LoRA-DAgger、Residual-DAgger、DSRL；每法 N=50 条额外 rollout，3 个种子 × 每点 25 次 rollout。附录 B 另有 Gr00t N1.7（LIBERO-90 task 57）与香草扩散策略（robomimic LIFT）。
6. **MetaWorld 主结果（§5.2，Table 1，π0.5，12 个任务）：** 平均成功率 Base 0.53、SFT 0.71、LoRA-DAgger 0.68、Residual-DAgger 0.64、DSRL 0.55、FlowDAgger 0.78（相对基座 +0.25）；论文称 FlowDAgger 在 12 个任务中胜 8 个。Door Lock 上 Residual-DAgger 0.85 高于 FlowDAgger 0.75（论文自述）。SFT 在 Box Close（0.70 对 0.59）、Bin Picking（0.76 对 0.69）等也更高，具体胜负见 Table 1。
7. **跨基座（§5.3，Table 2，7 个共享任务）：** π0.5 均值 0.53→0.79（+0.26），Cosmos-Policy 0.53→0.74（+0.21）。
8. **算力（§5.2，Fig. 4）：** 训练只更新小噪声策略，显存与部署 π0.5 所需的约 8 GB 相当，可在单张消费级 GPU 上训练；Fig. 4 图注给出 Assembly 成功率对训练峰值显存（8 / 22.5 / 70 GB 刻度）。
9. **保持先验（§5.4，Table 3）：** 在 Hammer 上适配（50 episode，SFT 为 50 demos 且匹配梯度步数），在 5 个近饱和的留出任务上评测：基座留出均值 0.96；FlowDAgger 0.88（−0.08），Residual-DAgger 0.69（−0.27），LoRA-DAgger 0.30（−0.66），SFT 0.02（−0.94）。Hammer 成功率：基座 0.40、FlowDAgger 0.84、SFT 0.80、LoRA-DAgger 0.68、Residual-DAgger 0.56。FlowDAgger 在 Push 上从 0.96 降到 0.56，在 Door 上从 1.00 降到 0.88。
10. **真机（§5.5，Table 4）：** 两个双臂平台（FR3 Duo、Dual UR5e），8 个任务，每格 30 次 rollout，只用 5 到 20 条干预 episode。Base / SFT / FlowDAgger：Block Pick 0.73/0.80/0.90（5 条）；Glassware Stacking 0.26/0.53/0.76（5）；Button Push 0.60/0.73/0.73（10）；Slider 0.36/0.43/0.66（5）；Wire Pull 0.40/0.53/0.70（10）；Jenga Stacking 0.76/0.90/0.86（5）；Toolbox Packing 0.13/0.63/0.80（10）；Plug Insertion 0.60/0.66/0.72（20）。论文称“除 Jenga 外均优于 SFT”，但表中 Button Push 为并列 0.73。§5.5 与 Table 4 没有明说真机所用基座策略是 π0.5 还是 Cosmos-Policy。
11. **反演方式对比（附录 B.1，Table 5，π0.5 + MetaWorld assembly，29 条脚本专家动作块）：** 动作 MSE：Euler 反推 0.0329、Adam 优化 0.0275、轨迹级不动点 0.0228、逐步不动点（M=5）0.00168；耗时 315 / 1350 / 1662 / 456 ms（RTX 5090）；下游 Assembly 成功率 0.59 / 0.79 / 0.77 / 0.87，Hammer 0.71 / 0.75 / 0.68 / 0.96。论文称误差大于约 0.03 MSE 的目标会损坏而非引导策略（Euler 反推下 Assembly 0.59 低于基座 0.64）。
12. **迭代数（附录 B.2，Table 6）：** 逐步不动点 M=3 / 5 / 10 的 MSE 为 0.00397 / 0.00168 / 0.00147，耗时 405 / 456 / 614 ms；轨迹级 k=5 与 k=20 的 MSE 为 0.0228 与 0.0237。
13. **其他基座（附录 B.3、B.4，单种子，每点 15 次 rollout）：** Gr00t N1.7 在 LIBERO-90 task 57 零样本 60%，FlowDAgger 约 20 episode 接近 100%，DSRL 平台约 70%；干预为脚本专家（scripted-expert）。扩散策略在 robomimic LIFT 零样本约 48%，FlowDAgger 约 25 episode 达约 97%，DSRL 约 80%。
14. **WAM 噪声策略参数化（附录 A.3）：** 联合噪声约 10^5 维，直接回归过高；对比“Full”（回归整个联合潜变量）与“Basis”（回归 k=64 个 PCA 系数，PCA 在基座预热阶段收集的反演噪声上拟合），两者都可行，Basis 更稳，是 Cosmos-Policy 实验的默认。
15. **论文自述局限（§6）：** 适配能力受预训练模型支持集限制，目标行为远离动作流形时反演只能恢复基座可表达的最近行为；依赖人类干预的质量与覆盖，系统性偏差会反映到策略；高度多模态动作分布或病态生成动力学下反演精度会下降。

## 对 wiki 的映射

- [FlowDAgger 详情页](../../wiki/entities/paper-rcl-2607-08877-flowdagger-human-in-the-loop-adaptation-of-gener.md)
- 清单摘录（非全文）：[rcl_awesome_wam_2607_08877_flowdagger-human-in-the-loop-adaptation.md](rcl_awesome_wam_2607_08877_flowdagger-human-in-the-loop-adaptation.md)
