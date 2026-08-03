# WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory（arXiv:2607.18840）

> 来源归档（ingest）

- **标题：** WorldScape Policy 2.0: Empowering Steerable World Action Modeling with Reasoning-Augmented Memory
- **缩写 / 框架：** **WorldScape Policy 2.0**（WS-Policy 2.0）；数据集 **ManipEvent-5M**
- **类型：** paper / world-action-model / manipulation / long-horizon / memory / multimodal-prompt
- **arXiv：** <https://arxiv.org/abs/2607.18840>（v1，cs.RO，Submitted 2026-07-21；正文日期 2026-07-20；PDF：<https://arxiv.org/pdf/2607.18840>）
- **项目页：** <https://manifoldai-research.github.io/WorldScape-Policy/> — 归档见 [`sources/sites/manifoldai-research-worldscape-policy.md`](../sites/manifoldai-research-worldscape-policy.md)
- **代码（论文/项目页声明）：** <https://github.com/manifoldai-research/WorldScape-Policy> — **占位仓库，代码未发布**；归档见 [`sources/repos/worldscape-policy.md`](../repos/worldscape-policy.md)
- **权重：** <https://huggingface.co/manifoldai-research/WorldScape-Policy-2> — **占位卡片，权重未发布**（声明 Apache-2.0）
- **作者：** Haisheng Su（Project Lead）、Zongdai Liu、Xin Jin、Haoxuan Dou、Chengming Hu、Baorun Li、Zhanwang Liu、Ruiyan Xu、Jianjie Fang、Xin Zhang、Zhenjie Yang、Xue Yang、Chen Gao、Junchi Yan、Yong Li、Wei Wu（通讯）
- **机构：** Manifold AI、清华大学、上海交通大学
- **入库日期：** 2026-08-03
- **一句话说明：** 在统一 video-action DiT 上同时挂 **短期视觉记忆（近 4 chunk 干净 latent 做 causal prefill）** 与 **长短期事件记忆（VLM 输出的 global-history / local-active / event-boundary 三视图检索 + 门控融合）**，用 **semantic forcing** 把事件级字幕语义蒸馏进隐式子目标规划通路；配套 **ManipEvent-5M**（4.89M 事件段 / 744K episode / 512M 帧）做事件级预训练，形成「高层指令自主规划 + 细粒度文本 / 目标图 / 演示视频可控执行」的统一接口。

## 开源状态（步骤 2.5）

- **项目页核查（2026-08-03）：** 项目页 Code / Model 两栏均给出链接（GitHub + Hugging Face），但打开后：
  - GitHub `manifoldai-research/WorldScape-Policy` 仅有 `README.md` + `.gitignore`，README 明写 *"Code is coming soon. We are preparing the training, inference, and evaluation code for release."*
  - HF `manifoldai-research/WorldScape-Policy-2` 仅有模型卡，明写 *"Model is coming soon. We are preparing the pre-training model and post-training checkpoint of RoboTwin 2.0 dataset for release."*；卡片声明 **Apache-2.0**。
- **结论：** **宣称将开源 / 待发布**（截至入库日既无可运行训练/推理入口，也无权重）。ManipEvent-5M 自采部分（PiPER 数据集、UMI 数据集）**未见发布计划**，公开来源部分（AgiBot World / RoboMIND / RoboCOIN / DROID / RoboTwin 2.0 / LIBERO / EgoDex）需各自单独获取。
- **复现读法：** 目前只能按论文重建；wiki 实体页不得写成「已开源」，`## 源码运行时序图` 按 **不适用** 处理。

## 摘录 1：三个痛点与总体主张（§1）

- **痛点 A｜记忆缺进度感：** 现有 WAM 或只条件于当前观测（静态记忆），或只用滑窗短历史；不同阶段视觉相似时（如某个中间子目标前后桌面几乎一样）会丢失「已经做到哪一步」。MemoryWAM 用混合记忆放开全历史访问，但**只是检索视觉证据**，没有把历史组织成任务进度、也没转成可用于自主规划的隐式子目标条件。
- **痛点 B｜语言–视频–动作接地太粗：** 多数具身数据集中同一 episode 的所有片段共享同一条任务级指令，对内部各原子动作是弱监督；模型容易走视觉捷径，生成看似合理的 video-action 轨迹但**对细粒度原子动作变化不敏感**。
- **痛点 C｜交互接口受限：** 现有 WAM 基本只能用文本控制，用户自然表达意图的方式（目标图、演示视频、跨本体示例）无法原生输入，限制视觉上下文推理与跨本体迁移。
- **核心洞见：** 长程可控性需要**两种互补的时间上下文**——**语义事件记忆**（推理任务进度）与**帧级视觉记忆**（保留局部交互动力学）。二者分别落在 VLM 分支与 causal DiT 分支，不混为一谈。

**对 wiki 的映射：** 新建 [`wiki/entities/paper-worldscape-policy-2.md`](../../wiki/entities/paper-worldscape-policy-2.md)；与 [World Action Models](../../wiki/concepts/world-action-models.md) 的 Joint WAM 族、[Worldscape-MoE](../../wiki/entities/paper-worldscape-moe-heterogeneous-action.md)（同 Manifold AI，偏上游可控视频 WM）互链。

## 摘录 2：Reasoning-Augmented Long Short-Term Memory（§3.1–§3.3）

- **建模目标：** \(p_\theta(A^{(e)}_t, z_{t+1:t+H} \mid o_t, a_{<t}, \mathcal{P}^\mu_t, \mathcal{M}_t)\)；\(\mathcal{M}_t=(\mathcal{Q}_t, \mathcal{Z}^{vis}_t)\) 即 **事件记忆队列 + 近期视觉 latent buffer**。
- **动作表示：** 单臂 10D = chunk 首帧相对位移 \(\Delta p\in\mathbb{R}^3\) + 相对旋转的连续 6D 表示 + **绝对** 1-DoF 夹爪；双臂 \(d_a=20\)。**embodiment 专属 action encoder/decoder** 只做接口，**flow matching 在原始动作空间**做（不是在 embedding 上）。
- **两种语言模式**（\(\mu\in\{\text{auto},\text{fine}\}\)，**互斥而非拼接**）：
  - `fine`：事件级字幕 \(c_t\) 经 **T5** 编码后直接 cross-attention 进 DiT；
  - `auto`：只给 episode 级指令 \(y\)，由 VLM 分支联合当前观测 + 事件记忆推断当前子目标，输出 **memory-enhanced VLM tokens**。
- **单次 VLM prefill 复用：** 感知 token \(u_t=H^{ctx}_t\) 与 \(K=4\) 个自回归 planning token 的末层隐状态 \(r^{1:K}_t\) 来自**同一条推理轨迹**（共享 KV cache），拼成当前 reasoning latent \(q_t=[u_t; r^{1:K}_t]\)。
- **事件记忆三视图**（历史 chunk 的感知 token 先 attention pooling 压成固定数量 gist token，planning token 全保留 → 紧凑历史 bank \(H_t\)）：

  | 视图 | 构造 | 作用 |
  |------|------|------|
  | **Global-History** \(m^{gh}\) | \(P_{gh}(\text{Expand}(F_{gh}([e_y;\text{Mean}(H_t)])))\) | 任务意图 + 全轨迹池化，概括整体进度 |
  | **Local-Active** \(m^{la}\) | 最近 \(S_e\) 个已完成 chunk 的**全 token anchor** | 当前活跃事件的细节 |
  | **Event-Boundary** \(m^{eb}\) | \(d_j=1-\cos(\bar q_j,\bar q_{j-1})\)，`TopKΔ` 取至多 \(S_b\) 个（带最小时间间隔 \(\Delta\)），保**全 token** | 稀疏保留子任务起止/切换点，**不需要在线边界标注** |

- **检索与门控：** \(B_t=\text{cat}(m^{gh},m^{la},m^{eb},H_t)\)（**保留全历史 bank，选择不丢弃非边界证据**）；\(q_t\) 作 Query 做 cross-attention 得 \(\tilde m_t\)，再逐 token 门控 \(\gamma_t=\sigma(W_g[q_t;\tilde m_t]+b_g)\)、\(\hat q_t=q_t+\alpha\gamma_t\odot\tilde m_t\)。
- **短期视觉记忆：** \(\mathcal{Z}^{vis}_t=z^{obs}_{t-S_v:t}\) 干净 VAE latent 作 DiT 前缀；可选目标图 / 演示视频 latent \(p^{vis}_t\) 作**持久前缀**（rollout 全程不滑动）。causal mask 允许未来 video-action token 看持久提示与干净历史，但禁止看未来 chunk。
- **默认容量：** 短期视觉记忆 **最近 4 个 chunk**；长期事件记忆 **默认 8 个历史 chunk**（可按任务时长调）。

**对 wiki 的映射：** 实体页画「双记忆 + 门控融合 + 模式选择」流程图；强调 **事件记忆走 VLM 分支、视觉记忆走 DiT 分支** 的分工，与 MemoryWAM 式「只检索视觉证据」区分。

## 摘录 3：Semantic Forcing 与三阶段课程（§3.4、§3.6）

- **Semantic forcing：** 训练时用同一事件的 T5 嵌入 \(s_t\) 做**语义靶**，对其归一化表示 **stop-gradient**（固定靶防止塌缩），只训投影器 \(\phi_q\) 把规划隐状态拉进 T5 语义空间：\(\mathcal{L}_{sem}=1-\tilde s_t^\top\tilde q_t\)。推理时细粒度字幕缺席，模型只能靠 \(\hat q_t\) 自己推子目标。
- **总损失：** \(\mathcal{L}=\mathcal{L}_{act}+\lambda_w\mathcal{L}_{world}+\lambda_s\mathcal{L}_{sem}\)，两个流匹配损失 + 语义对齐；**\(\lambda_s=0.001\)**（未启用时为 0）。多模态条件与记忆门控**端到端训**，不需要额外的提示/记忆标签。
- **三阶段课程：**

  | 阶段 | 训练内容 | 目标 |
  |------|----------|------|
  | **Stage 1** 事件级多模态 WAM 预训练 | ManipEvent-5M；细粒度字幕走 T5、目标图/演示视频走持久视觉前缀；**短期视觉记忆已启用**，无 VLM 事件记忆 | 建立细粒度语言–视频–动作接地与可控生成 |
  | **Stage 2** 记忆感知 mid-training | 引入 VLM 规划分支与长短期事件记忆；输入只给高层指令，事件字幕仅作 \(\mathcal{L}_{sem}\) 训练靶 | 把 Stage 1 的显式语义迁移进**自主规划**通路 |
  | **Stage 3** 下游交互式后训练 | 按任务所需模式（自主规划 / T5 细粒度 / 视觉提示）适配下游本体 | 统一接口下的任务特化 |

- **实现细节：** 标注用 **Qwen3-VL-32B**；隐式推理骨干用轻量 **Qwen3-VL-4B**，**只吃 head-view 图像、resize 到 320×160**；video DiT 由 **Wan2.2-5B** 文生视频权重初始化，video / action DiT **共享骨干**；三路相机拼成单张视觉画布；video-action token 间**双向注意力**，对干净/历史 token **保持因果**。预训练 bs 768 / lr 5e-4，后训练 bs 128 / lr 6e-5 / 50K steps。

**对 wiki 的映射：** 实体页写「semantic forcing 是把 fine 模式的显式语义搬进 auto 模式隐通路」的机制读法；\(\lambda_s=0.001\) 与 stop-gradient 是可复现关键细节。

## 摘录 4：ManipEvent-5M 构成与标注管线（§3.5，Table 1）

| 数据类型 | 来源 | 帧（M） | 时长（h） | Episode（K） | 事件段（M） | 段/episode | 单段比例（%） | 训练占比（%） |
|----------|------|--------:|----------:|-------------:|------------:|-----------:|--------------:|--------------:|
| 真机（自采） | Self-Collected PiPER | 54.00 | 506.60 | 35.82 | 0.56 | 15.6 | 0.5 | **45.0** |
| 真机（公开） | AgiBot World | 285.56 | 2644.08 | 160.15 | 1.16 | 7.2 | 0.2 | 32.53 |
| 真机（公开） | RoboMIND | 6.56 | 60.75 | 10.27 | – | – | 100.0 | 0.75 |
| 真机（公开） | RoboCOIN | 32.00 | 302.94 | 44.27 | – | – | 100.0 | 3.65 |
| 真机（公开） | DROID | 27.00 | 500.82 | 92.23 | – | – | 100.0 | 3.07 |
| 无机器人 UMI | Self-Collected UMI | 9.50 | 90.60 | 26.02 | 0.06 | 2.3 | 30.8 | 8.0 |
| 仿真 | RoboTwin 2.0 | 8.16 | 42.16 | 35.73 | 0.145 | 4.1 | 26.8 | 3.94 |
| 仿真 | LIBERO | 0.12 | 3.86 | 1.71 | 0.005 | 2.9 | 0.12 | 0.06 |
| Ego 人视频 | EgoDex | 89.24 | 831.00 | 338.23 | 2.96 | 8.8 | 18.8 | 3.0 |
| **合计** | | **512.14** | **4982.81** | **744.43** | **4.89** | **6.6** | – | – |

- **注意训练配比 ≠ 数据量占比：** EgoDex 贡献了 **2.96M / 4.89M（≈61%）** 的事件段，但训练占比只有 **3.0%**；自采 PiPER 只占 0.56M 段却吃掉 **45%** 训练配比——真机自采数据被显著上采样。RoboMIND / RoboCOIN / DROID **完全没有事件级分解**（单段比例 100%）。
- **动作规范化：** 机器人轨迹统一到 §3.1 的笛卡尔末端表示；**人手姿态重定向到机器人夹爪位姿与开合**，使人–机数据在同一 position–rotation–gripper 语义下联合预训练，本体差异交给专属 encoder/decoder。
- **四段式层级标注（Qwen3-VL）：**
  1. **两段式标注** — 先从本体归一化的平移/旋转速度、夹爪状态变化、episode 端点得到候选边界（短段合并 + 自适应粒度约束），赋 move / grasp / release / idle 粗原语；再让 VLM 做开放词表语义重标。**解耦「何时变」与「变的是什么」**。
  2. **层级帧采样** — episode 级采 ≤16 帧 + 有序事件表/边界/原语先验/执行结果，产出 `high_level`（**事后描述实际完成的任务**，而非照抄原指令，对缺失指令、执行偏差、重试更鲁棒）；事件级每段均匀采 ≤12 帧、收集可用视角，产出单动作 `step_text`。
  3. **具身提示约束** — 提示里强制写明动作顺序、执行末端、目标物体、初末状态、接触模式、运动方向、物体交互、身体运动、失败/恢复行为，避免「move the cup」这类泛化描述。
  4. **多视角冲突消解** — 按 `LEFT_GRIPPER → RIGHT_GRIPPER → HEAD` 的 local-first 顺序组织；**冲突时优先夹爪视角**（对接触/抓取状态/局部目标身份更可靠），缓解杂乱场景的目标物混淆。失败的事件字幕**局部替换**而不丢弃整条轨迹。
- **视觉提示构造：** 目标图分 **first-view**（该事件末帧，同相机同本体，无歧义后置条件）与 **third-view**（任务匹配的人示教在对应事件完成时刻，画面含演示者，要求策略抽象掉演示者与视角）；视频提示分 **human→robot**（自采真机轨迹配同任务 UMI 演示，按全局任务/有序事件/终态匹配）与 **robot→robot**（同语义任务不同本体，按有序事件描述与目标态对齐）。

**对 wiki 的映射：** 实体页单列「数据配比读法」——事件段数与训练配比的错位是本文数据工程的核心取舍；标注管线的「两段式 + 事后描述 + 夹爪优先」三条可迁移到其他事件级数据集构建。

## 摘录 5：实验结果（§4，Table 2–6 + Fig. 10）

**RoboTwin 2.0 标准协议**（50 任务 × 100 trial，所有方法均用 clean+randomized 数据微调 50K 步）：

| 类别 | 模型 | Clean | Randomized | Average |
|------|------|------:|-----------:|--------:|
| VLA | \(\pi_0\) | 65.9% | 58.4% | 62.2% |
| VLA | X-VLA | 72.9% | 72.8% | 72.9% |
| VLA | \(\pi_{0.5}\) | 82.7% | 76.8% | 79.8% |
| VLA | Abot-M0 | 81.2% | 80.4% | 80.8% |
| VLA | LingBot-VLA | 86.5% | 85.3% | 85.9% |
| VLA | HoloBrain-0-QW | 91.9% | 92.3% | 92.1% |
| WAM | Motus | 88.7% | 87.0% | 87.9% |
| WAM | GigaWorld-Policy | 85.6% | 85.3% | 85.5% |
| WAM | LingBot-VA | 92.9% | 91.6% | 92.3% |
| WAM | Fast-WAM | 91.9% | 91.8% | 91.9% |
| WAM | Abot-M0.5 | 94.0% | **94.2%** | 94.1% |
| WAM | LingBot-VA 2.0 | 93.8% | 93.4% | 93.6% |
| WAM | **WorldScape Policy 1.0** | 93.2% | 91.7% | 92.5% |
| WAM | **WorldScape Policy 2.0** | **94.3%** | **94.2%** | **94.3%** |

- **标准榜已接近饱和：** 领先 Abot-M0.5 仅 **+0.2**、LingBot-VA 2.0 **+0.7**；相对自家 1.0 **+1.8**，相对 \(\pi_{0.5}\) **+14.5**。
- **C2R（Clean-to-Randomized，只用 clean 演示训练、在 clean+randomized 两种评测上取平均，Fig. 10）**：\(\pi_0\) **31.4** / \(\pi_{0.5}\) **37.5** / Fast-WAM **39.1** / **WS 2.0 47.9**（分别 +16.5 / +10.4 / +8.8）。**真正拉开差距的是这条 OOD 口径。**

**真机（AgileX 双臂 PiPER，5 任务，各 20 trial，episode 级成功率）**：

| 能力 | 任务 | 提示 / 上下文 | \(\pi_{0.5}\) | DreamZero | WS 2.0 |
|------|------|---------------|-----:|-----:|-----:|
| 长程自主规划 | 叠衣服 | 全局指令 | 60% | 45% | **75%** |
| 长程自主规划 | 折箱子 | 全局指令 | 65% | 55% | **75%** |
| 记忆依赖视觉推理 | 猜盒子（shell game） | 演示视频 | 30% | 50% | **75%** |
| 跨本体技能迁移 | 叠积木 | 目标图 | 10% | 20% | **60%** |
| 跨本体技能迁移 | 叠积木 | 演示视频 | 20% | 20% | **70%** |
| 序贯细粒度控制 | 清桌（完整序列） | 连续子任务字幕 | 70% | 60% | **80%** |

> 注：\(\pi_{0.5}\) 原本只吃单帧静态观测，作者为 shell game 与演示条件叠积木**人为扩展了其输入历史**以保证可比。

**细粒度文本指令跟随（清桌，重置场景下逐条指令各 20 trial，Table 4）**：

| 划分 | 指令 | \(\pi_{0.5}\) | DreamZero | WS 2.0 |
|------|------|-----:|-----:|-----:|
| In-domain | 白纸巾 → 篮子 | 70% | 75% | **80%** |
| In-domain | 纸杯 → 篮子 | **90%** | 75% | **90%** |
| In-domain | 塑料瓶 → 篮子 | **90%** | 70% | **90%** |
| OOD | 黑笔 → 篮子 | 50% | 40% | **70%** |
| OOD | 绿胶带 → 篮子 | 55% | 35% | **60%** |
| OOD | 米色鞋 → 篮子 | 25% | 15% | **50%** |
| — | In-domain 均值 | 83.3% | 73.3% | **86.7%** |
| — | **OOD 均值** | 43.3% | 30.0% | **60.0%** |
| — | 总均值 | 63.3% | 51.7% | **73.3%** |

**消融（RoboTwin 2.0，官方 clean-only 训练划分；绝对值不可与 Table 2 直接比）：**

| Table 5 | STM | LTM | LSR | Clean | Randomized | Average |
|---------|:---:|:---:|:---:|------:|-----------:|--------:|
| 无记忆 | ✗ | ✗ | ✗ | 64.60% | 17.22% | 40.91% |
| +短期视觉记忆 | ✓ | ✗ | ✗ | 66.92% | 22.42% | 44.67% |
| +长期事件记忆 | ✓ | ✓ | ✗ | 68.49% | 24.01% | 46.25% |
| +隐式子目标推理 | ✓ | ✓ | ✓ | **69.74%** | **26.03%** | **47.89%** |

| Table 6 | Stage-1 | Stage-2 | SF | Clean | Randomized | Average |
|---------|:-------:|:-------:|:--:|------:|-----------:|--------:|
| 从零训 | ✗ | ✗ | ✗ | 65.67% | 20.71% | 43.19% |
| 仅 Stage-1 | ✓ | ✗ | ✗ | 67.90% | 25.36% | 46.63% |
| Stage-1+2 无 SF | ✓ | ✓ | ✗ | 68.95% | 25.64% | 47.30% |
| 完整课程 | ✓ | ✓ | ✓ | **69.74%** | **26.03%** | **47.89%** |

- **消融读法：** 记忆三件套在 **randomized** 上贡献 **17.22 → 26.03（+8.81）**，远大于 clean 上的 **+5.14**——记忆主要买的是 **OOD 鲁棒性**，不是干净场景的上限。三阶段拆解中 **Stage-1 事件级预训练增益最大**（尤其 randomized +4.65），Stage-2 与 SF 各自只再加约 **+0.67 / +0.59**。
- **口径提醒：** 消融完整模型 **47.89%** 与 Fig. 10 的 C2R **47.9%** 一致，即消融跑的就是 C2R 协议。

**对 wiki 的映射：** 实体页「结论」明确写两点——(1) RoboTwin 2.0 标准榜已饱和，本文头条增益 +0.2~+0.7 无选型意义；(2) 真正的差异在 C2R（+8.8 vs Fast-WAM）与真机的视觉提示 / 记忆任务（shell game 75 vs 30/50，叠积木 60–70 vs 10–20）。

## 摘录 6：未写明的边界（据全文核查）

- 论文**没有独立 Limitations 章节**，以下为按正文推得的边界，写 wiki 时需标注为「本页推断」：
  - **无任何延迟 / 吞吐数字**：每个 action chunk 都要跑一次 Qwen3-VL-4B prefill + 4 token 自回归解码，再走 Wan2.2-5B 级 DiT 流匹配；论文只说「为降显存只吃 head-view 320×160」，未报告闭环控制频率。
  - **记忆容量默认很小**：长期事件记忆默认只留 **8 个历史 chunk**，其「全历史」是压缩后的 gist bank 而非完整保留；论文未给出更长 horizon 下的扩展曲线。
  - **真机规模有限**：单一 AgileX 双臂 PiPER 平台、5 个任务、每任务 20 trial；跨本体迁移只在数据侧（human→robot / robot→robot 视频提示）验证，未在**第三方本体**上做闭环。
  - **基线可比性自述**：\(\pi_{0.5}\) 在两个视觉上下文任务上的输入历史由作者扩展；DreamZero / \(\pi_{0.5}\) 的后训练配置细节未展开。
  - **消融只在 clean 划分**：全部消融走 C2R 协议，未在标准 clean+randomized 协议下复核各组件贡献。

**对 wiki 的映射：** 实体页「常见误区或局限」照此写，并把「代码/权重未发布」列为首要复现风险。

## 建议 wiki 动作

- 新建 **[`wiki/entities/paper-worldscape-policy-2.md`](../../wiki/entities/paper-worldscape-policy-2.md)**（含流程总览 Mermaid、结论、`## 源码运行时序图` 标注不适用）。
- 在 **[`wiki/concepts/world-action-models.md`](../../wiki/concepts/world-action-models.md)** 的 Joint WAM 族补一条文献实例。
- 在 **[`wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md`](../../wiki/overview/wm-action-consequence-category-01-wam-action-prediction.md)** 本组工作表补一行。
- 在 **[`wiki/entities/paper-worldscape-moe-heterogeneous-action.md`](../../wiki/entities/paper-worldscape-moe-heterogeneous-action.md)** 补同团队上下游关系（上游可控视频 WM ↔ 下游 WAM 策略）。
