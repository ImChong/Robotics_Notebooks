# ASU CSE 598（Fall 2026）：Physical AI 客座讲座（Xinyuan Liu）

> 来源归档（ingest · 课程客座讲座官方课件）

- **标题：** Physical AI（ASU CSE 598 · Fall 2026）
- **类型：** course
- **讲者：** Xinyuan Liu（LENS Lab, ASU）
- **课程：** Arizona State University CSE 598（Fall 2026）
- **官方课件：** <https://drive.google.com/file/d/19M1BsdvSGGHaS0SpdblaL1DRg_5AfxSt/view?usp=sharing>（`26fall_CSE598_Physical_AI.pdf`，71 页）
- **讲课视频：** <https://youtu.be/p1KjuI2z8uM>
- **关联课程整理页：** [human five 课程笔记归档](../blogs/wechat_human_five_robot_policies_2026-11-24.md)
- **扩展清单仓：** <https://github.com/LENS-Lab-AI/Awesome-Frontier-Model-Robotics>（课件第 4 节引用）
- **入库日期：** 2026-11-25
- **版本：** Google Drive 公开 PDF `26fall_CSE598_Physical_AI.pdf`（共 71 页，文件 ID `19M1BsdvSGGHaS0SpdblaL1DRg_5AfxSt`）
- **核对范围：** 2026-11-25 通过官方 Google Drive 文件下载并提取全部分页文本（第 1–71 页），逐页通读全部 7 个主题板块（Foundations、Imitation Learning、Deep RL、Token VLAs、Diffusion & Flow Policies、World Models、Putting it together）及第 71 页 Key readings；讲者备注（speaker notes）与 YouTube 录像未单独观看或转录。
- **通读：** 已完成
- **一句话说明：** 围绕“pick up the red cup”贯穿示例，以统一策展目标 $\max_\theta \mathbb{E}_{(o,a)\sim q}[w(o,a)\log \pi_\theta(a\mid o)]$ 串起机器人控制接口、模仿学习、深度强化学习、Token VLA、Diffusion/Flow 策略与 World Models，并给出四层控制频率栈与真机故障定位表。

---

## 为什么值得保留

- **补全被中文二手笔记截断的后半程主线：** [human five 课程笔记归档](../blogs/wechat_human_five_robot_policies_2026-11-24.md) 仅覆盖第 1–4 节（到 Token VLA），官方课件第 5–7 节完整给出了 Diffusion Policy、Flow Matching、$\pi_0$–$\pi_{0.7}$ / GR00T 家族对比、World Models（Dreamer V3 / V-JEPA 2-AC / Cosmos 3 / DreamZero）及系统部署排查框架。
- **统一数学视角明确边界：** 第 7 页把 BC/ACT/DP/VLAs、DAgger、Filtered BC、PPO 与 $\pi^*_{0.6}$ Recap 并置在“数据分布 $q$ + 样本权重 $w$ + 策略族 $\pi_\theta$”三轴之下，清晰区分“训练阶段策略更新”与“部署阶段闭环执行”。
- **真机工程细节具体可核查：** 给出 Unitree G1 底层关节指令公式、Diffusion Policy 经典 $H=16, E=8, K=10\text{–}100$ 参数、四层控制频率栈与杯子打滑排查五问表。

## 课件结构与核心要点（第 1–71 页）

| 板块 | 页码 | 核心内容 |
|---|---|---|
| 引言与统一目标 | 1–9 | UMI（RSS 2024）抓红杯示例；策略闭环 $a_t\sim\pi_\theta(\cdot\mid o_t)$、$s_t$ 与 $o_t$ 区分；$J(\theta)$ 与 $p_\theta(\tau)$；$\max_\theta \mathbb{E}_{(o,a)\sim q}[w(o,a)\log\pi_\theta(a\mid o)]$ 对比 BC/ACT/DP/VLA、DAgger、Filtered BC、PPO、$\pi^*_{0.6}$ Recap |
| 1. Foundations | 10–17 | Joint vs Task 空间；Absolute（LeRobot 默认）/ Relative（相对于预测 chunk 起始状态，$\pi_0$）/ Delta（相对于上一步动作）；Position / Velocity / Impedance / Torque 控制模式与 Unitree G1 指令 $\tau=k_p(q^*-q)+k_d(\dot q^*-\dot q)+\tau_{ff}$；Sim2Real（DR、真机混训、Cosmos 重光照/纹理、SAGE + GapONet 逐关节误差测量） |
| 2. Imitation Learning | 18–29 | BC MSE 回归；Push-T 多模态平均失效；GMM/BC-GMM（robomimic）；BC-RNN；ACT（预测 100 步动作，训练 CVAE 推断 $z$、部署设 $z=0$、执行短前缀如 4/16 并重规划）；复合误差 $O(T^2\epsilon)$；DAgger 与 HG-DAgger；ALOHA / VR / SpaceMouse / UMI / Unitree VR 遥操作 |
| 3. Deep RL | 30–37 | Unitree Go2 MimicKit 小跑示例；Policy Gradient 与 Actor-Critic；PPO-Clip（$\epsilon=0.2$）；`legged_gym` 奖励项；AMP（SIGGRAPH 2021 + MimicKit G1 侧手翻）；LLM/VLM 自动奖励：Eureka（GPT-4 写奖励代码）与 ROOT（LENS Lab，VLM 评审 rollout 视频，$\max_\psi F(\pi^*_\psi)$ s.t. $\pi^*_\psi=\arg\max_\pi J(\pi;r_\psi)$） |
| 4. Token VLAs | 38–46 | LLM 与 VLA 对照；RT-1（~130k 真机轨迹）与 RT-2（256 离散桶，四舍五入损精度）；Open X-Embodiment（1M+ 轨迹、60 数据集、22 本体）；OpenVLA（7B Llama 2、DINOv2+SigLIP、970k 轨迹、7-D/步、LoRA）；FAST（DCT+Round+BPE 压缩）；OpenVLA-OFT（并行连续解码）；MolmoAct（depth tokens → visual trace → action tokens）与 MolmoAct2（连续动作头，课件报告优于 $\pi_{0.5}$）；高层规划器 **Gemini Robotics 2**（`ER 2` 规划并调用 VLA 技能） |
| 5. Diffusion & Flow Policies | 49–59 | Diffusion Policy（RSS 2023，预测噪声损失，$H=16, E=8, K=10\text{–}100$）；Flow Matching（线性插值 $X_\tau=(1-\tau)\epsilon+\tau A$、速度场回归、测试 ~10 步）；Octo（RSS 2024，非 VLM，800k 轨迹、25 数据集、27M/93M）；$\pi_0$（PaliGemma 3B VLM + ~300M flow action expert，50 步块，`openpi`）；$\pi$ 家族（$\pi_0, \pi_{0.5}, \pi_{0.6}, \pi^*_{0.6}\text{ Recap}, \pi_{0.7}$）；GR00T N1 / N1.6（开源 3B）/ N2（WAM）；离散 Token 头与连续动作头对比 |
| 6. World Models | 60–64 | 四种角色：Learned simulator（Dreamer V3）、Planner（V-JEPA 2-AC，1M+ 小时视频 + 62 小时机器人视频，$a^*=\arg\min_a\|P(z_t,a)-z_{\text{goal}}\|$）、Data maker（Cosmos / Genie 3）、World Action Model（DreamZero, GR00T N2）；Cosmos 1–2.5 到 Cosmos 3（2026-05：Reasoner + Generator 双塔开源模型，Super 32B & Nano 8B）；WAM 联合建模 $p(\text{video},\text{actions})=p(\text{video})\cdot p(\text{actions}\mid\text{video})$ 及串联 vs 联合（DreamZero） |
| 7. Putting it together | 66–71 | 2×2 方法旋钮矩阵（固定数据集 vs 真机 rollout × 专家示范 vs 奖励信号）；按“缺失原料”选型表（“Pick by the missing ingredient, not by what is newest”）；四层控制栈（Planner 秒级 → VLA/WAM ~5–50 Hz → Whole-body RL ~50 Hz → PD ~1 kHz）；杯子打滑 5 假设排查表；Key readings |

## 核对边界记录

- **与原微信二手笔记的关键核对结论：** 二手笔记第 4 节中提到的“GPT 系列做 planner 调用动作模块”在官方课件第 46 页实际示例为 **Gemini Robotics 2（`ER 2`）**；第 3 节的 **ROOT** 为讲者所在 LENS Lab 工作，以 VLM 审视 rollout 视频做奖励函数搜索。
- **未核对部分：** 课件第 71 页注明“Sources for every slide are in the speaker notes”，PDF 正文中仅保留主体文字与主要图表标签，未包含讲者备注与 YouTube 演示录像原声。

## 对 wiki 的映射

- [robot-policies-il-to-token-vla](../../wiki/overview/robot-policies-il-to-token-vla.md) — **课程总览主页面**（`original` 主源）
- [robot-learning-five-paradigms-taxonomy](../../wiki/comparisons/robot-learning-five-paradigms-taxonomy.md) — 范式与选型对比
- [imitation-learning](../../wiki/methods/imitation-learning.md)、[reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[vla](../../wiki/methods/vla.md)、[action-chunking](../../wiki/methods/action-chunking.md) — 对应核心方法页
- [diffusion-policy](../../wiki/methods/diffusion-policy.md)、[generative-world-models](../../wiki/methods/generative-world-models.md) — 第 5–6 节延伸方法页

## 参考来源（原始）

- Google Drive 官方课件：`https://drive.google.com/file/d/19M1BsdvSGGHaS0SpdblaL1DRg_5AfxSt/view?usp=sharing`
- YouTube 讲座录像：`https://youtu.be/p1KjuI2z8uM`
