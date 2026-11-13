---
type: entity
project_id: x-navdp
tags:
  - paper
  - navigation
  - diffusion-policy
  - reinforcement-learning
  - visual-navigation
  - cross-embodiment
  - isaac-lab
  - unitree-go2
  - unitree-g1
  - fudan
  - shanghai-ai-lab
  - sysu
  - tsinghua
status: complete
updated: 2026-11-12
arxiv: "2607.28560"
code: https://github.com/InternRobotics/NavDP
original:
  - ../../sources/papers/x_navdp_arxiv_2607_28560.md
related:
  - ../tasks/vision-language-navigation.md
  - ../overview/paper-notebook-category-08-navigation.md
  - ../../roadmap/depth-navigation.md
  - ./paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md
  - ./paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md
  - ./paper-roamflow.md
  - ./paper-occplanner-2608-14160.md
  - ./paper-rlinf-vla.md
  - ../methods/diffusion-policy.md
  - ../methods/ppo.md
  - ../concepts/diffusion-model.md
  - ../concepts/reward-design.md
  - ../concepts/sim2real.md
  - ./unitree-g1.md
sources:
  - ../../sources/papers/x_navdp_arxiv_2607_28560.md
  - ../../sources/sites/x-navdp-project-page.md
  - ../../sources/repos/internrobotics_x_navdp.md
  - ../../sources/repos/navdp.md
summary: "X-NavDP（arXiv:2607.28560，复旦/上海 AI Lab/中山/清华）：在 NavDP 上用同状态组内 Q 归一化的加权 score matching（GQRM）+ 自举轨迹扰动做跨本体在线 RL 后训练；仿真 40 个留出场景 SR 61.20→84.28，真机 9 格平均 10%→约 65%（每格 10 次）。"
---

# X-NavDP：跨本体导航扩散策略 RL 后训练

**X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching**（[arXiv:2607.28560](https://arxiv.org/abs/2607.28560)，v1 为 2026-07-30；[项目页](https://yty-sky.github.io/x-navdp-project-page/)；[HF 资产](https://huggingface.co/InternRobotics/X-NavDP)）作者来自复旦大学、上海人工智能实验室、中山大学、清华大学。本页依据 arXiv v1 全文逐段通读后改写，章节号指该论文。

## 一句话定义

在预训练的 [NavDP](./paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) 上，用「同一状态候选组内归一化 Q 值」作权重的 score matching 做在线 RL 后训练（GQRM），配合自举轨迹扰动探索与本体 FiLM 条件，使一个共享的 RGB-D 点目标导航扩散策略在轮式、四足、人形三种本体上同时提升（论文摘要，§1）。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| GQRM | Group Q-score Reweighted Matching | 同状态候选组内归一化 Q 作权重的 score matching（§4.3） |
| NavDP | Navigation Diffusion Policy | 被后训练的 IL 预训练骨干（privileged 信息引导） |
| DPMD | Diffusion Policy Mirror Descent | 奖励加权 score matching 的前作，GQRM 的基础（§3.2） |
| DPPO / DSRL | Diffusion Policy Policy Optimization / DSRL（论文称其优化初始扩散噪声，§2.2） | 论文表 3 中的对照 RL 微调法 |
| PMD | Policy Mirror Descent | KL 近端正则的策略改进目标（式 3 与 4） |
| FiLM | Feature-wise Linear Modulation | 用本体嵌入产生缩放平移参数（式 13） |
| RTC | Real-Time Chunking | 部署期时序一致性引导的灵感来源（§4.4） |
| MPC | Model Predictive Control | 把轨迹转成底层速度指令（附录 B.4） |
| SR / SPL | Success Rate / Success weighted by Path Length | 成功需到达目标 0.5 m 内（§5.1） |

## 为什么重要

- **动机（作者自述，§1）：** 模仿全局 oracle 轨迹，部署却只有局部 RGB-D，监督与观测不匹配，抑制探索与脱困；数据生成对本体不敏感，跨平台表现不稳。
- **扩散加 RL 的两条已有路线都有短板（作者自述，§1、§5.3）：** 策略梯度（DPPO）需经全去噪链算似然，不稳；潜空间法（DSRL）冻结主体，探索受限。表 3 在其 24 场景设置里给出 DPPO 33.98 与 DSRL 13.65 的 SR，均低于 SFT 的 53.50。
- **证据上最值得记住的一点：** 表 3 中同为 reweighted score matching 的 DPMD-original（无组内归一化）已有 71.57 SR，GQRM 为 80.48，差 8.9 个点；也就是说大部分增益来自「放弃似然梯度改用加权 score matching」这一选择，组内归一化是其上的增量（我的读法，来自表 3）。
- **可复现栈：** 代码在 [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP) 的 `baselines/x-navdp`，权重在 HF（见 [仓库归档](../../sources/repos/internrobotics_x_navdp.md)）。

## 核心原理与流程总览

三个组件（§4）：

1. **自举扰动（§4.2）：** 同一观测下取 goal 条件轨迹与 goal 无关（no-goal）轨迹，`τ_mixed = s ⊙ (τ_pointgoal + λ·τ_nogoal)`（式 7 与 8），s 以概率 ε 独立翻转整条 chunk 的 x 或 y 符号，得到侧移、倒退、绕行候选。λ 为带符号系数，允许外推到原 goal 条件分布之外。
2. **GQRM（§4.3、附录 A）：** 对同一状态 s 的 N 个候选，用双 critic 取小值 q，组内中心化后除以标准差（附录 B.2 写明除的是 minibatch 上的值离散度，与正文式 10 的组内 σ 表述略有出入），乘 c 并裁剪，得 `Q̃_G`；以 `exp(Q̃_G/λ)` 加权去噪 score matching（式 11）。只保留 `Q̃_G>0` 的前 k 个。
3. **本体条件（§4.4）：** 本体嵌入一方面加到动作 token（式 12），另一方面经 FiLM 调制解码后特征（式 13）。部署期另加 RTC 式引导（式 14 与 15）。

训练与控制栈（§4.1、附录 B）：IsaacLab 并行；扩散策略每次输出轨迹 chunk，MPC 跟踪 3 秒，本体专属步态策略执行；一个 3 秒窗的累计奖励记为一次 macro-step 转移。

```mermaid
flowchart LR
  obs[RGB-D + 点目标 + 本体 ID] --> pol[NavDP 骨干 + 本体 FiLM]
  pol --> goalc[goal 条件候选]
  pol --> nog[no-goal 候选]
  goalc --> mix[自举扰动 混合与符号翻转]
  nog --> mix
  mix --> roll[IsaacLab 多本体 rollout 经 MPC 与步态策略]
  roll --> buf[回放缓冲 macro-step 奖励]
  buf --> crit[双 critic 估 Q]
  crit --> gq[同状态组内归一化 取 top-k 正优势]
  gq --> upd[加权 score matching 更新最低噪声 6 步]
  upd --> pol
```

## 工程实践

以下均为论文附录 B 与 §5.1 的设置，除非另注：

| 项 | 论文设置 |
|----|----------|
| 场景 | GRScenes-100，56 训练（47 住宅 + 9 商业）/ 40 留出（20 + 20） |
| 并行 | 每场景 1 个进程 × 8 环境 = 448 环境；本体平均分给 Dingo / Go2 / G1，首次 30,000 步后切换，之后每 24,000 步切换 |
| 优化 | Adam，学习率 2e-5 线性降到 1e-5；γ=0.99，τ=0.005；DDP 同步；每轮每进程 192 条转移分 8 个 minibatch |
| 候选 | 每状态 64 个候选取 5 个最高；值裁剪 [-3, 3]；10 步去噪中只微调最低噪声 6 步 |
| 训练日程 | 3000 步回放预热，再 100 次仅 critic 更新，之后 critic 每次更新、actor 每两次更新一次（B.2） |
| 奖励 | 六项：存活惩罚、A* 路径进度、到达、碰撞、停滞、G1 倒地惩罚（附录 B.1，表 4） |
| 控制 | 非线性 MPC，|v|≤0.5 m/s，|ω|≤0.5 rad/s，3 秒跟踪（B.4） |
| 耗时 | 摘要与 §1 称后训练仅 12 小时（所用 GPU 数论文正文未明示，未核实） |
| 代码状态 | 仓库 MIT；Isaac Sim 与 Isaac Lab 的具体版本、acados 依赖、`SCENE_DIR` 布局等来自此前对 HF README 的核查，本次未重读，复现前以官方 README 为准 |

## 评测证据与清单说法的对比核对

对象是 Glen Berseth 的 CoRL 2026 阅读清单对本文的一句话摘要：「RL post-training of navigation diffusion policies with group Q-score reweighted matching, generalizing to new embodiments and behaviors.」清单的排序由作者个人兴趣模型打分，摘要为 LLM 据摘要起草，并非论文质量评价。

| 清单用语 | 论文中的情况 |
|----------|--------------|
| RL post-training of navigation diffusion policies | 属实：在 NavDP 上做在线 RL 后训练（§4）。但对象是「一个」预训练的 NavDP，并非泛指任意导航扩散策略；论文没有换别的骨干做验证 |
| group Q-score reweighted matching | 属实，是方法名（§4.3）；清单未提另两个组件（自举扰动、本体 FiLM），表 6 显示去掉探索设计会让 SR 掉到 6.23，扰动并非可有可无 |
| generalizing to new embodiments | 需限定：「跨本体」指同时在 Dingo、Go2、G1 三种本体上联合训练并在同样三种本体上评测（§5.1、附录 B）；并没有测试训练中未见过的新机器人形态。作者在局限中自认：换新形态需要对应的预训练底层控制器（§6）。真机的轮式平台为 Turtlebot，与仿真的 Dingo 不同，算是一处小的本体差异，但论文未单独分析 |
| generalizing to new behaviors | 部分属实：新行为（脱困后退、绕长障碍、选更安全路径）仅由图 3 定性轨迹展示（§5.2），我未能看到该图，也没有对应的定量「新行为」指标；表 6 间接显示反向翻转探索对 SR 影响很大 |
| 未提数字 | 论文的主要数字：仿真 SR 61.20→84.28、SPL 58.95→77.19（表 1）；真机九格各 10 次，NavDP 平均 10%，X-NavDP 平均约 64.4%，摘要写 65%（表 2）；清单未涉及这些，也未提真机每格仅 10 次 |

## 结论

X-NavDP 支持对生成式导航策略进行跨本体后训练，但低层控制器、短记忆与感知缺口仍限定部署范围。

- 复现时核对组内 Q 归一化、轨迹扰动和各本体控制接口。
- 分开报告 24 场景消融与 56 场景结果，并增加真机每格十次之外的重复。
- 针对透明障碍、窄通道和长障碍绕行另设验收，新形态仍需对应的预训练步态控制器。

## 局限

**论文自己承认的（§5.3、§6）：**

- 只依赖短时上下文，长时记忆不足，绕长障碍时可能沿原路返回。
- G1 窄通道通过困难；透明、镂空障碍（玻璃隔断、穿孔板）感知不可靠。
- 只在三种本体、预定义步态控制器上验证；新形态需要对应预训练底层控制器。
- 真机评测的场景、环境条件与平台范围有限。

**我看到的未验证之处（推测与读法，非作者自述）：**

- 真机每格仅 10 次，没有置信区间，单格相差 10 个百分点就是 1 次成败；NavDP 在 lab 与 office 场地全为 0，说明这些场地对基线可能极难，但论文并未说明这些场地与「hard case」的选取标准（摘要用了 hard cases，§5.1 只写 lab、hall、office）。
- 论文称真机包含室内外环境（§6），但 §5.1 列出的三个场地是 lab、hall、office，哪些属于室外未说明。
- 表 3 至 7 的消融在 24 场景子集上进行，只有表 1 与表 5、7 的完整行用 56 场景；24 场景与 56 场景结果不可直接并列。
- 表 5 中 24 场景下 FiLM 与 Soft-Prompt Concat 仅差 0.24 SR，FiLM 的优势主要在 56 场景下才拉开（84.28 对 82.84），单次运行，未报告随机种子方差。
- 表 1 中 NavOL、SIDP、NavDP-RL 只有部分格有数字，且来自各自论文，协议是否对齐未核实；X-NavDP 在 RL 阶段使用的训练场景与评测来自同一 GRScenes-100 数据集的不同场景划分，分布较近。
- 主文式 10 的 σ 是组内标准差，附录 B.2 说「按 minibatch 上的值离散度归一化」，两处描述不完全一致，实现以代码为准（未核对代码）。

## 关联页面

- [NavDP（PNB）](./paper-notebook-navdp-learning-sim-to-real-navigation-diffusion.md) — 被后训练的预训练骨干
- [NoMaD（PNB）](./paper-notebook-nomad-goal-masked-diffusion-policies-for-navigat.md) — 扩散导航的先行工作
- [RoamFlow](./paper-roamflow.md) — 另一条 image-goal 导航路线
- [OccPlanner](./paper-occplanner-2608-14160.md) — 另一条导航扩散规划工作
- [RLinf-VLA](./paper-rlinf-vla.md) — 论文引用的大规模 RL 框架 RLinf 的 VLA 方向相邻工作
- [diffusion-policy](../methods/diffusion-policy.md)、[diffusion-model](../concepts/diffusion-model.md) — 扩散策略与去噪基础
- [PPO](../methods/ppo.md) — 论文提到 PMD 与 TRPO、PPO 的关系（§3.2）
- [reward-design](../concepts/reward-design.md) — 附录 B.1 六项奖励的设计参照
- [sim2real](../concepts/sim2real.md) — 本文真机为零微调部署
- [unitree-g1](./unitree-g1.md)、[vision-language-navigation](../tasks/vision-language-navigation.md)、[导航类论文总览](../overview/paper-notebook-category-08-navigation.md)、[depth-navigation 路线](../../roadmap/depth-navigation.md)

## 参考来源

- [x_navdp_arxiv_2607_28560.md](../../sources/papers/x_navdp_arxiv_2607_28560.md) — 论文全文通读归档
- [x-navdp-project-page.md](../../sources/sites/x-navdp-project-page.md)
- [internrobotics_x_navdp.md](../../sources/repos/internrobotics_x_navdp.md)
- [navdp.md](../../sources/repos/navdp.md)

## 继续阅读

- [项目页](https://yty-sky.github.io/x-navdp-project-page/)
- [arXiv PDF](https://arxiv.org/pdf/2607.28560)
- [Hugging Face 模型卡与安装说明](https://huggingface.co/InternRobotics/X-NavDP)
- 前作 DPMD：Ma 等，Efficient Online Reinforcement Learning for Diffusion Policy（arXiv:2502.00361，论文参考文献 [16]）
