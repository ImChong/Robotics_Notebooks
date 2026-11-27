# Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement

- **作者：** Mingtong Zhang, Dhruv Shah
- **机构：** 普林斯顿大学（Princeton University）
- **版本 / 载体：** arXiv:2606.18247v1 [cs.RO]（2026-06-16 提交）；项目页及官方代码仓库标注为 Proceedings of Robotics: Science and Systems (RSS) 2026
- **原文：** [arXiv:2606.18247 (HTML v1)](https://arxiv.org/html/2606.18247) / [arXiv 摘要页](https://arxiv.org/abs/2606.18247) / [官方项目主页](https://veritas-improvement.github.io)
- **项目页 / 代码：** 已开源 — 官方项目主页 [veritas-improvement.github.io](https://veritas-improvement.github.io)，官方代码仓库 [princeton-prism/veritas](https://github.com/princeton-prism/veritas)（MIT 许可证，默认分支 `main`，完整包含 `eval_with_veritas_verifier.sh`、`src/agent/eval.py`、`src/agent/train.py`、三种视觉验证器实现与 Bridge/SimplerEnv 配置；注：arXiv HTML 首页 `\code` 链接存在拼接笔误 `github.com/prism-princeton/projectgithub.com`，实际有效仓库为 `princeton-prism/veritas`）
- **核对范围：** 通读 `arXiv:2606.18247v1` HTML 全文（Abstract、§1 Introduction、Fig. 1、§2 Related Works、§3 Preliminaries、§4 Generator-Verifier Framework §4.1–§4.4、Algorithm 1、Fig. 2、§5 Instantiating VERITAS §5.1–§5.3、§6 Experiments §6.1–§6.5、Fig. 3–9、§7 Discussion、References [1]–[52]、Appendix §8 Implementation Details §8.1–§8.2、Algorithm 2–5、Table 1–2、Appendix §9 Additional Experiments / Fig. 10–11），同步核读官方项目主页内容（`veritas-improvement.github.io`）及官方 GitHub 仓库 [princeton-prism/veritas](https://github.com/princeton-prism/veritas) 的文件清单与 `README.md`。
- **通读：** 已完成

## 原文核对笔记

- 对 wiki 的映射：[Visual Verification Enables Inference-time Steering and Autonomous Policy Improvement](../../wiki/entities/paper-agentic-2606-18247.md)
- 代码库静态核对归档：[princeton-prism/veritas](../repos/paper-agentic-2606-18247.md)

### 1. 研究动机与生成器-验证器框架（§1, §3, §4）

- **核心问题：** 通用机器人策略（如 $\pi_0$）严重依赖昂贵的人类遥操作示范数据，规模呈线性人力成本受限；现有测试时搜索（test-time scaling）方法多依赖额外训练的价值函数/验证器或单步局部微调，且只带来“一次性”推理收益，未形成持久化策略改进闭环。
- **VERITAS 范式（Generator–Verifier Framework）：**
  - **随机生成器（Stochastic Generator）：** 利用预训练流匹配/扩散型通用策略 $\pi_\theta$ 的多模态分布先验，在每个决策步 $t$ 并行采样 $N$ 个短时域动作块（action chunks，时域长度 $H$）：$\bm{a}_{t:t+H}^{(i)} \sim \pi_\theta(\cdot \mid o_t, l),\ i \in \{1,\dots,N\}$。在动作块层级而非单步动作层级采样，既摊销验证成本，又赋予验证器评估行为后果的时间上下文。
  - **无梯度视觉验证器（Gradient-Free Visual Verifier）：** 定义即插即用标量评分函数 $V(o_t, \bm{a}_{t:t+H}, l) \in \mathbb{R}$，无需更新策略参数或训练专用推理模型。
  - **Best-of-$N$ 推理时引导（Phase 1）：** 计算 $v_i = V(o_t, \bm{a}_{t:t+H}^{(i)}, l)$ 并执行最高分候选 $\bm{a}_{t:t+H}^\star$。
  - **自主离线策略改进飞轮（Phase 2）：** 将验证器引导下成功完成的轨迹汇集为 $\mathcal{D}_{\text{auto}}$，以标准监督行为克隆（BC）损失 $\theta' \leftarrow \arg\min_\theta \mathbb{E}_{\mathcal{D}_{\text{auto}}}[-\log \pi_\theta(\bm{a}_{t:t+H}^\star \mid o_t, l)]$ 微调原策略，把推理时计算蒸馏进策略权重。由于数据由策略自身产生，天然满足 on-policy 与运动学可行性，有效缓解模仿学习中的分布偏移（covariate shift）。

### 2. 三种视觉验证器设计与实现细节（§5, §8.1–§8.2）

- **VERITAS 主验证器（Algorithm 2，两阶段静态像素轨迹验证）：**
  1. **回合初轨迹生成：** 在任务起始帧 $F_0$（$o_0$）与自然语言指令 $\mathcal{I}$ 下调用前沿 VLM（Gemini 2.5）：首先识别任务相关目标并分配语义角色（`source`、`target` 等），结合检测/跟踪模型（代码中使用 Grounded-SAM-2：Grounding-DINO + SAM 2）提供图像空间位置提示；随后 VLM 一次性输出 5–10 个绝对像素路径点 $\mathcal{W} = \{w_1, \dots, w_K\}$（起止点不可跳过，每点含 `uv`、`tol_px`、`min_hold`、`skippable`、`weight`）。
  2. **在线几何评分：** 执行时无需逐帧调用 VLM 或在线目标追踪。将候选动作块预测的末端执行器（EEF）轨迹投影到图像平面 $\{e_t\}$，计算到当前激活路径点的像素欧氏距离 $d_t = \|e_t - \mathcal{W}[i].uv\|$；满足 $d_t < \text{tol}_i$ 且持续 $\ge \text{min\_hold}$ 帧（代码另含迟滞驻留与跳过机制）则推进路径点 $i \leftarrow i+1$；进度 $p_t = i / |\mathcal{W}|$，归一化评分 $s_t = \alpha p_t + (1-\alpha)\exp(-d_t/\tau)$（完成时 $s_t = 1.0$）。在线验证耗时 $< 1\text{ ms}$，在单张 RTX 5090 上以 $N=5$ 保持 15 Hz 控制频率。
- **消融对比的另两种验证器（§8.1）：**
  - **VLM-Constraints Verifier（Algorithm 3）：** VLM 不输出绝对像素坐标，而输出基于物体角色的相对引用（如 `role:source`、`role:target`、`midpoint(refA,refB)`），执行期间每帧通过物体跟踪器动态解析为像素目标并按物体尺度归一化距离 $d_t = \|e_t - g_i\| / \text{scale}(g_i)$。
  - **Heuristic Verifier（Algorithm 4 & 5）：** 由 VLM Router 识别任务类型与追踪物体后，提取 EEF–物体距离、相对运动与夹爪状态等几何特征，驱动 7 阶段有限状态机（`APPROACH -> ALIGN -> ENGAGE -> MANIPULATE -> RELEASE -> SETTLE -> DONE`）并计算进度分，停滞时可选调用 VLM 诊断故障。
- **微调超参数（Tables 1 & 2, §5.3）：**
  - $\pi_0$-Bridge（仿真）：开放执行时域 4 步，全局 batch size 1024，单卡 batch size 16，学习率 $5 \times 10^{-5}$，weight decay 0，max grad norm 1.0。
  - $\pi_0$-DROID（真机）：开放执行时域 8 步，batch size 256，学习率 $5 \times 10^{-5}$，warmup 1000 步，LR decay 1,000,000 步，微调 20,000 步；部署时除腕部相机与第三视角外，增设标定正视相机以减少遮挡。

### 3. 实验结果与关键发现（§6, §9）

- **推理时引导（§6.2）：** 涵盖 3 种基础策略（$\pi_0$-Bridge、$\pi_0$-DROID、$\pi_{0.5}$-DROID）及总计 1,160 次评估回合。无需参数微调，VERITAS 在 SimplerEnv 4 项仿真 WidowX 任务（`put eggplant in basket`、`stack cube`、`carrot on plate`、`spoon on towel`）上平均成功率提升 **12.6%**，在真机 DROID 4 项任务上平均提升 **35%**；三种验证器均在仿真中平均优于离线 RL 价值函数基线 **V-GPS**，而纯视觉原语提议基线 **PIVOT** 在真机上无法完成任务。
- **自主策略微调（§6.3）：** 在仿真中收集 **656** 条验证器筛选成功轨迹微调 $\pi_0$-Bridge，在 960 个评估回合上平均成功率提升 **9.7%**（最难任务 `Stack Blocks` 从 **31.3%** 提升至 **59.2%**，增幅 **+27.9%**）。
- **自主数据 vs. 人类遥操作数据效率（§6.4, Fig. 7）：** 在真机 4 项任务（`Pick up mouse`、`Insert marker`、`Carrot on plate`、`Put tape into wooden box`）上分别用 20、30、50、100 条轨迹微调（每组评估 20 次真机试验）。在 `Pick up mouse` 上 50 条自主轨迹优于 50 条人类轨迹（**0.65 vs. 0.60**）；在 `Carrot on plate` 上 50 条自主轨迹达到 **0.70 vs. 0.65**（§6.4 正文；100 条时两者收敛一致）；在 `Insert marker` 上全预算区间差距均在 **5 个百分点**以内。
- **超参数敏感性与局限（§6.5, §7, §9）：** 采样数 $N$ 超过 8 后收益趋于饱和；开环执行至多 10 步未出现显著退化；受限于初始静态像素轨迹（高动态场景适应性弱）、推理采样带来的计算开销（线性随 $N$ 增长）以及基础策略预训练先验的覆盖上限。对比 **RoboMonkey**（Appendix §9, Fig. 10）表明，单步高斯局部扰动仅优化局部行为且易强化策略原有偏差，而 VERITAS 在动作块层级的比较才能区分语义不同的策略路径。
