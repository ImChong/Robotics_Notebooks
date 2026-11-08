# What Can RL Bring to VLA Generalization? An Empirical Study（arXiv:2505.19789）

> 来源归档（paper）

- **标题：** What Can RL Bring to VLA Generalization? An Empirical Study
- **类型：** paper
- **作者：** Jijia Liu*、Feng Gao*、Bingwen Wei、Xinlei Chen、Qingmin Liao、Yi Wu、Chao Yu†、Yu Wang†（清华大学深圳国际研究生院 / 交叉信息研究院 / 电子工程系；*同等贡献，†通讯）
- **发表：** NeurIPS 2025（论文页脚 "39th Conference on Neural Information Processing Systems"）
- **arXiv：** <https://arxiv.org/abs/2505.19789>
- **PDF：** <https://arxiv.org/pdf/2505.19789>
- **项目页：** <https://rlvla.github.io>（论文摘要给出，本次未打开）
- **入库日期：** 2026-11-08
- **版本：** 阅读的是 arXiv v4（2026-01-14，页眉 `arXiv:2505.19789v4 [cs.LG] 14 Jan 2026`）；v4 含附录 B.4 真机、B.5 动作分块、B.6 开关门任务，早期版本可能没有这些内容（推测，未对照 v1）。
- **核对范围：** 下载 v4 PDF，pdftotext 提取后通读正文 §1–§6（含摘要、图 1–9 的图注与图中可提取文字、局限）与附录 A（A.1 算法实现、A.2 运动规划器与 SFT 数据、A.3 资产、A.4 各评测任务定义）、附录 B（B.1 温度、B.2 LoRA 秩、B.3 表 1 全表、B.4 真机表 2、B.5 动作分块表 3 与图 13、B.6 开关门表 4）、附录 C；参考文献仅浏览。图 3b（PPO / GRPO / DPO 学习曲线）、图 4b、图 5、图 6 等曲线图没有可提取的数值，只按图注与正文文字叙述引用，未目视核对曲线数值。未运行代码，未读项目页与 GitHub 仓库。
- **通读：** 已完成
- **一句话说明：** 在 ManiSkill 仿真的 pick-and-place 上，以 OpenVLA（LoRA r=32）为基座，对比 PPO / GRPO / DPO 与 SFT：PPO 是三种 RL 算法里最有效的；PPO 微调相对 SFT 在语义与执行维度的分布外泛化明显更好，视觉维度大体持平。

## 核心摘录

### 1) 问题与设置（§1、§3、§5.1）
- 动机：VLA 主要靠 SFT（行为克隆），对分布偏移下的复合误差敏感（§1，引 Ross & Bagnell 2010 的 horizon 二次回报损失）；想系统回答 RL 相对 SFT 到底带来哪种泛化收益（§1）。
- 任务建模为语言条件 POMDP；策略输入最近 H 步观测与指令（§3.1）。
- **基座：** OpenVLA（SigLIP + DINOv2 视觉编码 + Llama-2 7B），单张 RGB 图 + 指令，H=1；动作按 RT-2 方式离散为 256 个 bin 并覆盖 Llama-2 词表中最少使用的 256 个 token，交叉熵只算动作 token（§3.2、图 2）。
- **LoRA：** 所有微调方法（RL 与 SFT）均用 LoRA，rank=32，加到 OpenVLA 全部线性层，value head 全参数训练（§4.1、附录 A.1）。
- **环境：** ManiSkill，8 自由度 WidowX-250S；观测 640×480 RGB + 指令；输出末端笛卡尔 delta + 二值夹爪；稀疏奖励：抓取并持续握住正确物体 0.1，成功放置 1.0（§5.1）。
- 只做 pick-and-place（"put $O$ on $R$"）作为主任务（§1、§5.1、附录 A.4）。

### 2) 三个泛化维度与测试项（§5.1、附录 A.4、图 1）
- 训练随机化：16 个桌面外观、16 个物体、物体与盘子位置在矩形区域内随机（附录 A.4）。
- **Vision：** Unseen Table（5 种新桌面）；Dynamic Texture（前景纹理叠加，weak 透明度 0.3 / strong 0.5）；Dynamic Noise（整图纹理叠加，weak 0.3 / strong 0.5）；纹理取自 16 种。
- **Semantics：** Unseen Objects（9 个新物体）；Unseen Receptacles（16 个新容器）；Unseen Instruction Phrasing（16 种新指令模板）；Multi-Object（两个物体选一，seen / unseen 各一版）；Distractive Receptacle；Multi-Receptacle（两个未见容器）。
- **Execution：** Unseen Position（物体与容器位置范围扩大）；Unseen Robot Init Pose（机器人初始关节随机）；Mid-Episode Object Reposition（第 5 步把物体传送到桌面新随机位置）。
- 摘要口径：测试时至少一个因素分布外；新增 9 个物体、16 个容器、5 个桌面、16 个干扰纹理（§5.1）。

### 3) 三种 RL 算法比较（§4.1、图 3、附录 A.1）
- PPO：on-policy，clipped 比值 + GAE；动作概率为各动作 token 概率之积；另测 PPO-ORZ（关 GAE，γ=λ=1）（§4.1、A.1 式 1–2）。
- GRPO：outcome supervision，同一初始状态采样一组轨迹；32 组 × 每组 8 条轨迹；成功回合在首次成功步截断，失败回合不截断；outcome 奖励为抓取 + 保持 + 放置奖励之和，仅在回合末给（A.1）。
- DPO：采用 Zhang et al. 2024b 的轨迹级 TPO；只用本文稀疏奖励推断偏好，不用原文的自评分与 LLM 稠密奖励（§4.1、A.1）。
- 结果（文字）：PPO 一致优于 GRPO，也优于 DPO；每个设置 2 个随机种子（§4.1、图 3b）。图中给出 PPO、PPO(ORZ)、GRPO、DPO-iter1、DPO-iter2 曲线，横轴到 1M 环境步（图 3b，仅从图内文字读到，未读曲线值）。
- **论文自己的解释（均为假设，措辞为 "we attribute / we hypothesize"）：**
  - 对 GRPO：自然语言任务与机器人任务的环境动力学根本不同；在机器人 POMDP 里，每个动作都会以非平稳方式依次改变环境状态，可能使 GRPO 的优势估计不稳定（§4.1）。
  - 对 DPO：稀疏奖励使不同轨迹的优劣难以区分；离线数据集与交互执行之间有显著分布偏移（§4.1）。
- 论文没有做消融去证明这两条解释；致谢中提到有人对 GRPO 调参的建议 "显著提升了工作质量"（Acknowledgements），说明 GRPO 做过调参，但调参细节正文未给。

### 4) PPO 高效配方（§4.2、图 4–5）
- 总体：主实验在单张 NVIDIA A100 上约 42 小时收敛（§4.2）。
- **共享 actor-critic 骨干：** actor 与 critic 共享整个 Transformer，三层 MLP value head 取最后一层 Transformer 在第一个动作 token 位置的隐状态 h0；h0 比末位 token 隐状态 hn 与拼接 [h0..hn] 回报更高更稳；独立 critic 骨干回报相当，但训练慢 35%、显存多 83%（81.3 GB 对 44.4 GB）（§4.2、图 4）。
- **VLA 预热：** 官方 OpenVLA 检查点（OXE 预训练）在本基准上表现差，用 140 条由 Octo-Small 与运动规划器收集的演示做预热；预热后达到收敛所需环境步约少 50%，足够交互后渐近回报相当（§4.2、图 5a）。
- **最小 PPO epoch：** epoch 超过 1 没有回报或样本效率增益，但墙钟时间近似线性增加，故固定 epoch=1（§4.2、图 5b/5c）。
- 其他：采样温度 1.0（温度过大阻碍训练，附录 B.1）；LoRA 秩越低训练略快，选 32 以保证容量（附录 B.2，未系统研究）。

### 5) RL 与 SFT 的泛化比较（§5.2–§5.4、表 1）
- SFT 数据规模：专家轨迹从数百到 64k 条（约 126 万 transition），约 16k 条处进入平台期，取 SFT-16k 为 SFT 基线（§5.2、图 6a/6b）。
- RL 在约 0.4M 环境步后超过 SFT-16k 的 OOD（物体 / 桌面）表现；收敛时训练分布上与 SFT-16k 相当，未见物体与桌面上高 42.6%（§5.3）。
- 表 1（成功率，SFT / RL，3 种子）：IND 0.781 / 0.938；Obj. 0.453 / 0.714；Instruct 0.672 / 0.891；M-Obj.(OOD) 0.297 / 0.578；Obj. Pos. 0.568 / 0.807；Robot Pose 0.339 / 0.797；Obj. Rep. 0.286 / 0.745；视觉项 Table 0.719 / 0.844、Texture-s 0.557 / 0.630、Noise-s 0.505 / 0.667。M-Recep. 0.458 / 0.599（附录 B.3 表 1）。
- 论文结论口径：Vision 大体相当、Semantics 明显更好、Execution 显著更好（§5.3、图 1、图 7）。
- 解释（均为假设）：视觉上 RL 与 SFT 都没有超出训练时施加的视觉随机性诱导出额外鲁棒性；语义上 RL 通过试错学到与物体类型较无关的 "抓取" 技能；执行上 RL 的训练轨迹覆盖更宽的工作空间与末端姿态，而 SFT 轨迹贴着数据里的运动规划路径（§5.3、§5.4、图 8）。
- 定性（§5.4、图 9）：强噪声下 SFT 抓起后反复掉落、RL 完成放置；未见物体时 SFT 反复尝试抓已拿起的物体而卡住；执行类测试中 RL 能从失败抓取与物体中途移位里恢复，SFT 不顾位置误差继续执行，论文推测是演示中没有这类情形。

### 6) 附加实验（附录 B.4–B.6）
- **真机（B.4，初步）：** 把 WidowX 换成 Franka Panda 并对齐相机，同一流程重训 RL 与 SFT 后零样本上真机，30 次试验：抓取成功率 SFT 0.10 / RL 0.43；pick-and-place 成功率 SFT 0.00 / RL 0.27（表 2）。
- **动作分块（B.5）：** OpenVLA-OFT，chunk=4，400 条无动作过滤的演示预热；PPO 调整为按动作维度裁剪、裁剪比 0.2→0.1、γ=0.96、λ=0.85；RL 仍优于 SFT（表 3、图 13），但 M-Recep. 为 SFT 0.365 / RL 0.297，RL 反而更低。
- **开关门（B.6）：** Franka，8 个训练铰接物体；RL 在语义与执行项优于 SFT（Execution Avg. 0.300 / 0.398），但 IND 为 0.745 / 0.724、Dynamic Noise 与 Re-phrased Inst. 上 RL 更差（表 4）。

### 7) 论文自述局限（§6）
1. SFT 仅用运动规划器生成的演示，可能不覆盖人类数据的多样性；2. 评测限于 pick-and-place（附录 B.6 另有一个开关门任务）；3. 主要实验均在仿真，真机验证仍是后续工作。

## 对 wiki 的映射

- [paper-rl-vla-generalization-empirical-study](../../wiki/entities/paper-rl-vla-generalization-empirical-study.md)
