# Claude plays robotics（Anthropic Frontier Red Team）

> 来源归档

- **标题：** Claude plays robotics / How Claude performs on robotics tasks
- **类型：** site / research blog（官方评测报告，非 arXiv 论文）
- **来源：** Anthropic Frontier Red Team
- **作者：** Shmuel Berman, Michael Ilie, Jia Deng, Daniel Freeman
- **链接：** https://www.anthropic.com/research/claude-plays-robotics
- **发布日期：** 2026-07-09
- **入库日期：** 2026-08-28
- **一句话说明：** 无机器人预训练的通用语言模型，在 **控制接口抽象层级** 不同时，机器人能力可差一个数量级：直接力矩几乎失败，监督预训练策略 / 写控制器则能完成有限导航与操作。
- **开源状态：** **宣称将开源 / 截至入库日未公开** — 文内写代码将落 `github.com/safety-research/embody`（`EXPERIMENTS.md` / `METRICS.md`）；2026-08-28 对该 URL 请求返回 **404**。见 [embody 仓占位](../repos/safety-research-embody.md)。
- **项目页：** 无独立 `*.github.io`；以本研究报告页为入口。
- **沉淀到 wiki：** [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)、[Embody 评测套件](../../wiki/entities/anthropic-embody.md)

---

## 抓取说明

- 以 **2026-08-28** 对 Anthropic Research 页公开 HTML 正文抽取为准。
- 图表分数以原文图注为准；本页只摘可操作结论，不转存全部柱状图数值。
- 仿真直接控制实验 **暂停物理步进等待 LLM**，测的是「若推理足够快」的上界，不是今日 API 延迟下的实时力矩环。

---

## 评测设定（摘录）

| 轴 | 内容 |
|----|------|
| **模型** | 12 个，5 家：Claude Opus 4 / 4.1 / 4.5 / 4.6 / 4.7、Mythos Preview；GPT-5.4 / 5.1；Gemini 3.1 Pro Preview / 2.5 Pro；Kimi K2.6；Qwen 3.6+ |
| **具身** | 经典控制玩具；仿真 Unitree Go2（12-DoF）与 G1（29-DoF）；固定基座 Franka Panda；**真机 Unitree Go2**（Project Fetch） |
| **仿真** | MuJoCo；VLA 与 RL 单元用 GPU；渲染 `osmesa` |
| **控制接口** | ① 直接控制（力矩/末端增量）② 程序控制（写 Python `controller(obs)->action`）③ 强化学习监督（改 reward/架构/日程后训 PPO）④ 高层策略（摇杆步态 / MolmoAct VLA 监督） |
| **任务族** | 经典控制；低层/高层 locomotion；低层/高层 manipulation（LIBERO 厨房场景） |
| **Harness** | Claude 走 Agent SDK（关掉内置工具，只暴露机器人动作服务）；其它经 OpenRouter 适配器 |

### 四条控制接口（文内定义）

同一任务可走完全不同的「模型完成方式」：

1. **Direct control**：逐步输出低层动作（力矩、力、或 7 维末端运动）。
2. **Programmatic / code control**：写 Python 控制器，运行时映射观测→动作。
3. **RL supervision**：模型定义 reward、网络与训练日程，调用 batched PPO（默认 `envapi/training_bridge.py:train_ppo_batched`，`BatchedEnvWarp`），再部署学到的策略。
4. **Policy control**：向预训练策略发高层命令（自然语言或速度/偏航）；操作侧用 **MolmoAct** 提案，LLM 接受/修改/替换 7 维臂动作。

---

## 核心发现（编译）

1. **分数取决于身体 × 接口，几乎与「模型本身」同等重要。** 同一模型在直接力矩上像废物，在高层策略上可以完成导航/抓放。人形尤其只能在高层接口上拿到牵引力。
2. **代际进步不均匀：** 高层接口最稳定变强；低层直接控制也在涨，但不一致。
3. **Locomotion：** 新模型在四足站立/平衡/程序步行上有有限全身控制；人形从倒塌姿态 **无人成功站起**；平衡（已站立）从 Opus 4 到 4.7 有可测进步。真机/仿真高层导航能做简单寻的，但 **空间记忆、自定位、长开环计划** 仍系统性失败。
4. **Manipulation 低层：** 接触与抓取中间阶段明显进步；**整任务成功仍罕见（约 0–5.5%）**。Mythos Preview 虽触达/抓取更少，但整任务成功率高于 Opus 4.6（后者调整过多）。
5. **Manipulation 高层（监督 MolmoAct）：** 远强于直接控制；但几乎所有模型仍 **弱于 VLA 单独跑**。最好的监督者能收回大部分损失；在 VLA 不会的新场景上，最强模型已能提供净增益。过早覆盖 VLA（Mythos Preview）会伤分。
6. **感知瓶颈主要是朝向，不是换视角：** 操作侧 **cursor（夹爪相机上可查询的红 X）** 把 10 任务子集成功率从约 6% 拉到 32%（Mythos）；导航侧 **罗盘（世界系朝向度数）** 全面抬分。深度热图 / 分割 overlay 近似中性或略伤。第三人称相机只帮 Opus 4.7 与 Mythos。把像素换成 Gemini 3.1 的文字描述会 **伤** 新模型、**帮** 旧模型——新模型已经在吃原始图像。
7. **额外推理预算几乎不救低层机器人：** 多数 Claude 代际在 locomotion / 操作上差异落在误差内；Mythos 在高层 locomotion 上是例外（adaptive-low 40.2 → adaptive-max 54.1）。
8. **能从经验学，但几乎只在短时程：** 经典控制的代际优势来自 **重试** 而非首试。截断操作上下文（保留前 10 轮 + 最近 N 轮）多数模型不掉分，有的还更好（context rot）。`oneshot_course` 有练习会涨，难课程练 20 次仍 0 成功——学的是特定序列，不是通用规划。
9. **延迟：** 无推理文本轮约 2–8 s，带图 5–15 s；高推理 15–60 s（尾部 60–180 s）。腿式实时约需 **83 Hz**，当前非推理推理约 **0.2–0.4 Hz**，差约两个数量级。臂没有平衡约束，未暂停仿真。
10. **真机 Go2：** 与仿真一致的视觉/空间失败（停在 1 m 外、走廊回路全失败、玻璃门反射当目标、准星误判垃圾桶位置）。Grok 4.1 Fast 曾冲向玻璃门反射中的桌子。

### 高层 locomotion 十一任务（文内表）

`find_x`、`visual_search`、`color_sequence`、`return_home`、`procedural_maze`、`invisible_walls`、`obstacle_course`、`oneshot_course`、`drift_detection`、`turn_correction`、`explore_report`。

复合分 0–100；Opus 4.1→4.5 与 4.7→Mythos 两跳；4.5–4.7 平台是平均假象（失败模式在换：闭环自校正变强、遮挡下重规划变弱）。

### RL 监督

几乎所有模型训 RL 策略弱于写 Python 控制器。TwinFlipper 上 GPT-5.4 是唯一稳定学出胜任策略的。G1/Go2 四小时 GPU scaffold 上 GPT-5.4 与 Mythos 最强。策略上限 20 万参数；单次训练不得超过会话 1/3。

---

## 安全含义（原文结论）

VLM 的真实世界影响力可随 **信息/工具/控制接口** 差几个数量级。评估与部署必须把 **访问级别** 当作系统的一部分。今日 frontier 模型 **没有预训练策略就不能控人形**；但通用聊天模型已能在好的一次运行里自己写工具、慢慢走过迷宫或把盘子放到炉上。建设侧：调试失败、监督已有控制器、生成训练数据。安全侧：带明确边界的物理访问。

---

## 对 wiki 的映射

| 主题 | 目标 wiki |
|------|-----------|
| 控制接口抽象层级决定能力 | `wiki/concepts/llm-robotics-control-interfaces.md` |
| Embody 评测套件实体 | `wiki/entities/anthropic-embody.md` |
| 高层 = 监督预训练 VLA / 步态 | `wiki/methods/vla.md`、`wiki/concepts/foundation-policy.md` |
| 推理 vs 控制频率 | `wiki/concepts/control-inference-frequency-decoupling.md` |
| 写控制器 / RL 监督 | `wiki/methods/aspire.md`、`wiki/methods/reinforcement-learning.md` |
| LIBERO 作为操作评测床 | `wiki/entities/libero-benchmark.md` |
| 访问级别与安全过滤 | `wiki/concepts/safety-filter.md` |
| 短时程 in-context 适应 | `wiki/concepts/robot-in-context-learning.md` |
| Go2 / G1 具身 | `wiki/tasks/locomotion.md`、`wiki/entities/unitree-g1.md` |

## 参考链接

- <https://www.anthropic.com/research/claude-plays-robotics>
- 宣称代码镜像：<https://github.com/safety-research/embody>（入库日 404）
- LIBERO：<https://github.com/Lifelong-Robot-Learning/LIBERO>
- MolmoAct（Allen AI，评测所用 VLA）：见 Hugging Face / AllenAI 发布页
