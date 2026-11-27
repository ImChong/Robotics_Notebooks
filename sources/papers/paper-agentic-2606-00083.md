# From Demonstrations to Rewards: Test-Time Prompt Optimization for VLM Reward Models

- **作者：** Christian Gumbsch, Leonardo Barcellona, Lennard Schünemann, Platon Karageorgis, Andrii Zadaianchuk, Zehao Wang, Sergey Zakharov, Fabien Despinoy, Rahaf Aljundi, Efstratios Gavves
- **机构：** University of Amsterdam；Catholic University of Leuven（KU Leuven）；Toyota Research Institute（TRI）；Toyota Motor Europe
- **版本 / 载体：** arXiv:2606.00083v1 [cs.LG]（2026-05-22 提交）；RLC 2026 论文（OpenReview: `wjuZ5Qy141`）
- **原文：** https://arxiv.org/abs/2606.00083 （PDF: https://arxiv.org/pdf/2606.00083 ）
- **项目页 / 代码：** 已开源（提示词优化主仓 https://github.com/cgumbsch/Demo2Reward ；下游 RL 实验姊妹仓 https://github.com/cgumbsch/Demo2Reward_RL ；OpenReview: https://openreview.net/forum?id=wjuZ5Qy141 ；仿真演示与预训练 BC 策略包通过 Google Drive 提供，真机数据集未随仓附带）
- **核对范围：** 全文通读 arXiv:2606.00083v1 正文第 1–5 节（Abstract、Introduction、Related Work、Method 3.1–3.3、Experiments 4.1–4.6、Discussion & Limitations）、图 1–7 与表 1–2，以及附录 A（A.1–A.5 超参数与提示词模板）、B（B.1–B.2 仿真与 Franka 真机细节）、C（C.1–C.4 附加结果、发现的指令原文、VLM-SD 与 RoboReward 消融及图 8–10）；同时核对 `cgumbsch/Demo2Reward` 与 `cgumbsch/Demo2Reward_RL` 两仓 README 与文件树。
- **通读：** 已完成

## 原文核对笔记

- **核心动机（奖励黑客 / Reward Hacking）：** 零样本 VLM 奖励模型（`VLM-SD`）在未精细调优提示词时往往产生高频假阳性（例如在 PickPlace-Can 或 Assemble-Nut 刚抓起物体但尚未放置完成时提前输出 `1`），导致下游策略通过反复触发假阳性状态刷取高合成奖励而真实任务成功率为 0（Sec. 3.2, Fig. 2）。
- **核心算法（Demo2Reward）：**
  - 利用策略学习前已采集的少量专家示范 $\mathcal{D}$（$N=3$ 条仿真轨迹或 $N=10$ 条真机轨迹），沿时间切分构建前缀子序列集合 $\mathcal{D}_{\text{sub}}$（包含正负样本，评测时统一均匀下采样为 5 帧、分辨率 $224\times 224$）。
  - 采用双 VLM 架构：固定 Critic $m_{\text{critic}}$（`Qwen3-VL-8B-Instruct`）输出二值成功判定 `{0, 1}`；Meta-Critic $m_{\text{meta}}$（`Qwen3-VL-8B-Instruct` 与 `Qwen3-VL-32B-Instruct`，采样参数 `temperature=1.3, top-p=0.97, top-k=50`）根据当前指令 $p$、一条完整参考示范、混淆矩阵指标（TPR/TNR/FPR/FNR）与最多 5 条误判片段迭代重写指令 $p'$。
  - 优化目标为 $\hat{\mathcal{O}}(p) = \text{TNR}(p) + \lambda\,\text{TPR}(p)$（$\lambda=0.01$，满分 $1.01$），优先消除假阳性（FPR→0）；为规避“全输出 0”的退化局部最优，每个 Meta-Critic 模型独立运行 $K=5$ 次（共 10 轮）、每轮 $I=100$ 步并选取最高得分提示词。
  - **测试时零额外开销：** 优化完全在策略训练前离线完成；在线 RL 仅在回合截断（truncation）时查询一次固定提示词的 Critic（禁用 GT 早停防标签泄漏）。
- **核心实验数据：**
  - **混淆矩阵（Tables 1–2, PickPlace-Can）：** 优化前 TPR=100%、FPR=26.1%、TNR=73.9%；优化后 FPR 降至 **0%**（TNR=**100%**），TPR=68.8%；Sec. 4.2 指出全部任务优化后的指令在演示集上均实现 100% TNR（0% FPR）。
  - **MetaWorld（IBRL, 10 seeds, Fig. 5）：** 覆盖 Assemble-Nut、Box-Close、Coffee-Push、Stick-Pull；稀疏回合末 GT 奖励峰值约 60%–80%，`RoboReward` 与 `RoboCLIP` 接近失效，`Demo2Reward` 优于所有 VLM 基线并带来最高 **+40%** 成功率提升（Coffee-Push 逼近 GT）。
  - **RoboMimic（IBRL, 5 seeds, Fig. 4）：** 在 PickPlace-Can（750k 步）和 NutAssembly-Square（1.5M 步）上显著优于零样本 `VLM-SD`。
  - **跨算法与提示词鲁棒性（Sec. 4.5, Fig. 6, 5 seeds）：** 切换为 `RLPD` 骨干及不同随机种子生成的提示词（p1/p2/p3）均保持稳定增益。
  - **真机实验（Sec. 4.6 & Suppl. B.2, Fig. 7, 3 seeds）：** Franka Research 3 + ZED 2i + GELLO + RLInf（RLPD）执行 `Lid-On-Pot`（30 步截断，20 条遥操作示范中取 10 条用于 Demo2Reward）；约 100 回合后成功率相比 `VLM-SD` 提升 **+20%** 并接近人工位姿阈值 GT 奖励。
  - **消融发现（Suppl. C.3–C.4）：** `VLM-SD` 采用 8B 模型 + 5 帧均匀采样 + 仅输出单字符 `{0,1}`，比完整视频、单末帧、输出推理过程（Descr.）、1-shot ICL 或 32B 模型效果更好且推理吞吐提升约 **10 倍**；`RoboReward` 无论归一化、非归一化（1–5）还是二值化（4/5→1）在 MetaWorld 均未取得有效学习，推测源于真机微调分布与仿真的域差。
- **已知局限（Sec. 5）：** 依赖少量固定视角示范，视角或光照变化易过拟合，且发现的提示词可能依赖 VLM 自身视觉感知偏差（如 Stick-Pull 将蓝色木棍描述为红色木棍）；当前仅支持二值稀疏奖励，且仍依赖人工给定初始任务指令 $\ell$。
- **对 wiki 的映射：** [Demo2Reward 实体页](../../wiki/entities/paper-agentic-2606-00083.md)
