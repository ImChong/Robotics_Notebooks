# RoboReact（arXiv:2608.03387）

> 来源归档（ingest）

- **标题：** RoboReact: Agentic Skill Distillation from Generated Egocentric Videos for Generalizable Whole-Body Manipulation
- **类型：** paper / humanoid / whole-body manipulation / video generation / VLM agent / skill distillation
- **arXiv：** <https://arxiv.org/abs/2608.03387>
- **PDF：** <https://arxiv.org/pdf/2608.03387>
- **HTML：** <https://arxiv.org/html/2608.03387>
- **项目页：** <https://roboreact.github.io/>
- **作者：** Shuliang He、Shuai Wang、Bo Yue、Junchi Teng、Changyu Wang、Guiliang Liu（通讯）
- **机构：** 香港中文大学（深圳）；京东科技；清华大学
- **入库日期：** 2026-08-14
- **一句话说明：** 单帧 egocentric RGB-D + 语言指令 → 生成人类交互视频 → 物体中心关键帧技能 → 冻结 VLM 用标定 rollout 做有界结构化编辑 → 测试时去掉 VLM，靠物体位姿再接地 + HOMIE 全身控制器在 G1 上执行长程双臂操作。无遥操作、无人类示教。

## 摘要级要点

- **问题：** 人形全身操作技能采集贵（遥操作 / 人体动作重定向 / RL 微调）。视频生成模型能从单帧想象人类操作，但如何把非度量生成视频变成高 DoF 人形可执行技能，此前几乎空白。
- **主张：** 不要回放连续轨迹。从生成视频抽出保几何的手–物关键帧，重定向到人形，再在标定 rollout 上用冻结 VLM 做 in-context 精炼；部署时技能冻结、VLM 不进控制环。
- **输入：** 任务提示 \(P_{\mathrm{task}}\)、初始 RGB-D \((I_1,D_1)\)、预训练视频生成模型 \(\mathcal{V}\)。
- **技能表示：** 有序关键帧 \(\Pi=\{(\rho_k,o_k,T^{l}_{a,k},T^{r}_{a,k},h^{l}_{k},h^{r}_{k},m_k)\}_{k=1}^{K}\)，阶段 \(\rho_k\in\{\mathrm{approach},\mathrm{align},\mathrm{fixed}\}\)。
- **优化：** 不走梯度。目标 \(\min_{\Pi\in\mathcal{F}}\lambda_s\mathcal{L}_{\mathrm{sem}}+\lambda_g\mathcal{L}_{\mathrm{geo}}+\lambda_m\mathcal{L}_{\mathrm{mot}}\)，由冻结 VLM 出 keep / align / offset / insert / delete，经 \(\mathrm{Proj}_{\mathcal{F}}\) 投影到可行技能。
- **执行：** WildDet3D 估物体位姿；当前 \( \hat{T}_{o_k} \) 把物体相对变换再接地为末端目标；HOMIE 跟踪基座/身高/躯干/臂/手命令。
- **平台：** 29-DoF Unitree G1 + BrainCo Revo2 Touch；头戴 RealSense D435i；外置 D435 第三人称标定；RTX 4080 Super 跑感知与高层。
- **任务：** Hand Over / Pour Water / Open Box / Open Drawer（长程双臂，20 trial / 任务）。
- **主结果：** 四任务终端成功率均值 **81.3%**，与 one-shot 真人视频先验 **80.0%** 持平；优于 ReKep 与 YOTO。冻结栈在蹲姿/物体平移/基座位姿扰动下保留名义 Avg. Len. 的 **80–94%**。
- **开源（截至 2026-08-14）：** 项目页与 PDF **未列训练/推理仓**；GitHub 用户 `RoboReact` 仅有落地页仓 `RoboReact.github.io` → **确认未开源**。未承诺 “code will be released”。

## 核心摘录（面向 wiki 编译）

### 两阶段：编译–精炼 vs 冻结执行

| 阶段 | 谁在环里 | 产出 |
|------|----------|------|
| Skill distillation | 视频生成 + VLM 选片 + 编译 + 标定 rollout + VLM 编辑 | 可行关键帧技能 \(\bar{\Pi}^\star\) |
| Test-time | 只估物体位姿并再接地；**无 VLM** | HOMIE 跟踪的全身命令 |

这是本文最该记住的架构：大模型只在「编译期」花钱，实时环只做几何再接地。

### 关键帧三阶段

| \(\rho_k\) | 含义 | 再接地 |
|------------|------|--------|
| approach | 接触前靠近 | 物体条件净空偏置 |
| align | 抓取/接触 | 默认用物体相对 \(\Delta T^{\diamond,*}_k\) |
| fixed | 相对几何锁死（如持物移动） | 机器人基座系命令，不绑可见物体锚点 |

### 精炼预算与编辑器能力（论文 Table 2–3）

- 0 round：几乎完不成任务（Hand Over Avg. Len. 0.08）。
- 15 rounds：Hand Over / Pour Water 各 **11/13** 终端完成。
- GPT-Codex **5.6-ultra** 比 **5.1-mini** 更能把 rollout 证据变成有效编辑（10 round 时 Pour Water SR +23.1、Open Box +30.8 点）。

### 消融读法（Pour Water）

| 去掉 | Avg. Len. | 读法 |
|------|-----------|------|
| 语义关键帧选择（改均匀采样） | 3.69 | 任务阶段结构丢了 |
| rollout 记忆 | 3.92 | 当前失败连不上历史证据 |
| 第三人称相机 | 4.77 | 总量影响小，但倾倒步从 12/13 掉到 6/13 |
| 完整系统 | 5.62 | — |

上游视频生成器：Seedance 2.0 优于 1.5 Pro；Open Drawer 对先验质量更敏感（接触丰富铰接）。

### 与基线的差（Table 1，终端 SR %）

| 方法 | Hand Over | Open Box | Pour Water | Open Drawer |
|------|-----------|----------|------------|-------------|
| ReKep | 35 | 15 | 40 | 20 |
| YOTO | 75 | 65 | 80 | 75 |
| One-Shot Real Prior | 85 | 70 | 80 | 85 |
| RoboReact | **85** | **70** | **85** | **85** |

生成视频先验略优于真人视频先验（均值 81.3 vs 80.0）。价值在任务顺序与手–物结构，不在度量轨迹。

### 标定期人类提示边界

每条技能蒸馏最多 **5** 条稀疏自然语言提示，只描述可观察失败、**不许给策略编辑**。测试评估禁止人类输入。不要把「无示教」读成「标定零人工」。

## 对 wiki 的映射

- 沉淀实体页：[RoboReact](../../wiki/entities/paper-roboreact.md)
- 交叉补强：[Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)、[ExoActor](../../wiki/methods/exoactor.md)、[Video-as-Simulation](../../wiki/concepts/video-as-simulation.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)、[HOMIE](../../wiki/entities/paper-loco-manip-161-040-homie.md)、[合成视频人形任务](../../wiki/entities/paper-synthetic-video-humanoid-tasks.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2608.03387>
- 项目页核查：[roboreact-github-io.md](../sites/roboreact-github-io.md)
