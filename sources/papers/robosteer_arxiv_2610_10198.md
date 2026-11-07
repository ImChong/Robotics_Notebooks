# RoboSteer 论文来源归档

## 元数据

- **题目：** Benchmarking Behavioral Steerability in Behavior Foundation Models（行为基础模型的行为可操控性基准评测）
- **基准：** RoboSteer
- **作者：** Minghe Gao、Zhanxi Yan、Jiahui Liu、Wendong Bu、Xiaoting Chen、Qizhou Wang、Yi Su、Siliang Tang、Jun Xiao、Yueting Zhuang、Tat-Seng Chua、Juncheng Li（通讯）
- **机构：** 浙江大学（ZJU）、宇树科技（Unitree Robotics）、新加坡国立大学（NUS）、中南大学（CSU）；部分研究在浙大—宇树具身智能联合实验室完成
- **预印本：** <https://arxiv.org/abs/2610.10198>
- **HTML 全文：** <https://arxiv.org/html/2610.10198v2>
- **项目页：** <https://robosteer.github.io/>（另见 [项目页归档](../sites/robosteer_project.md)）
- **数据：** <https://huggingface.co/datasets/YanCORANV/RoboSteer>（另见 [数据集归档](../repos/robosteer.md)）
- **版本：** arXiv:2610.10198v2，HTML 页眉日期 2026-10-11，cs.RO（初稿 2026-10-07）
- **核对范围：** v2 HTML 全文，正文 §1–§6、参考文献、附录 A–E（含 Tables 1–2、B.5–B.8 与评测定义），并核对官方项目页和数据集卡；2026-11-07
- **通读：** 已完成
- **代码：** 论文 §5.1 表示模型适配与评测代码将发布；截至 2026-11-07 官方项目页可用网页交互式评测器与 HF 权重，但尚未列出完整可运行的官方代码仓库。不能把网页评测服务等同于开源代码。

## 研究问题与贡献

传统行为基础模型（BFM）研究多以动作生成质量为核心指标，无法回答模型是否**按用户指定的方式**执行：例如移动方向、速度、动作次数、非主要肢体保持静止，以及有时序的跨模态指令。论文把“行为可操控性”（behavioral steerability）定义为生成行为对用户意图的忠实实现，并提出 RoboSteer 三层基准及 BG / IR / BS 统一记分方式。RoboSteer 是评测框架，**不是**新训练出的通用 BFM。

## 数据与任务构造（§4、附录 A–B）

- **规模：** 38,522 条独立源动作，77,044 个真人/骨架视频序列，共 91.58 小时；总计 **773,716** 条任务实例、20 个任务族；源动作关节数据对齐 Unitree G1 29 DoF，50 Hz。
- **Level 1 — Conditional Steering：** 12 任务族、630,623 实例（81.51%）；完整条件复现、时间补全、空间补全。条件包括文本、语音、真人或骨架视频、音乐、关节旋转、关键帧、局部目标。
- **Level 2 — Constraint Steering：** 7 任务族、135,329 实例（17.49%）；Speed、Amplitude、Direction、Order、Times、Trajectory、Body Restrain。Order/Times 使用额外的 BABEL/AMASS 标注片段，不是仅由 38,522 条主源动作产生。
- **Level 3 — Compositional Steering：** 1 任务族、7,764 实例（1.00%）；文本/语音/视频及边界关键帧按时间交错，包含 33,814 个模态片段/边界条件。
- **构造链：** 互联网人体视频 → 时空切分与多条件过滤 → Gemini-3-Flash-Preview 等生成动作语义与人体重建 → GMR 物理约束下的 G1 重定向 → SONIC 跟踪/可执行性筛选 → 任务标注与多模态条件构建。语音由 Edge-TTS 合成。
- **文件组织：** `Tasks/Level{1,2,3}/` 存任务 JSON；`Data/Level{1,2,3}/` 存任务资产；`Data/Shared/` 存复用动作与视频。动作包包含同步的 `joint_pos.csv`、`joint_vel.csv`、`body_pos.csv`、`body_quat.csv`、`body_lin_vel.csv`、`body_ang_vel.csv`。

## 指标与具体边界（§4.3、附录 C）

- **生成质量 BG：** `BG = 1000 * exp(-0.3 * FID - 1.6 * MM-Distance)`。两个系数经验标定，跨模态 MM-Distance 依赖各模态冻结评测器；另外报告多样性、Contact Sliding、R@K、MPJPE、速度误差、节奏指标等。
- **操控性 BS：** `BS_l = BG * product(IR_k, k=1..l)`，`IR_k` 在 0–1 之间。该乘积是论文**对同一评价对象**的形式化定义；Table 2 各层报告用不同任务子集聚合，不能从汇总均值直接推定层间逐行单调下降。
- **Level 1 的 IR₁：** 完整复现任务设定 `IR₁=1`（所以 `BS₁=BG` 属定义，不是异常）；时间补全看生成/参考时长的 tIoU；空间补全看缺失关节的角度范围重叠（JRO）；关键帧/目标到达看 G1 正运动学的姿态位置误差。
- **Level 2 的 IR₂：** 七类约束各计算通过率 `CG_c`，再**宏平均**。Speed 只判生成时长比参考更短/更长，**不验证是否精确达到指令中的 2× 或 0.5× 倍率**；Amplitude 比较关节最大角度范围的增减方向；Direction 用相对起始航向的横向位移并允许 ±60° 角误差；Body Restrain 要求限制区域 RMS 链接速度 ≤0.10 m/s、其他区域 ≥0.12 m/s。Order/Times 需视频 VLM 判断。
- **Level 3 的 IR₃：** 各时间段的意图实现分按时长加权平均；BS₃ 在框架层面与前两层 IR 因子相乘。

## 实验结果（正文 Table 2；2026-10-11 v2）

数值保留两位小数，按论文不同输入模态分别报告 BG / BS-L1 / BS-L2 / BS-L3：

| 输入 | 基线 | BG | BS-L1 | BS-L2 | BS-L3 |
|---|---|---:|---:|---:|---:|
| Text | GEM + GMR | 114.09 | 68.38 | 16.15 | 42.12 |
| Text | LoM + GMR | 104.63 | 66.27 | 16.83 | 39.28 |
| Text | MotionCraft + GMR | 114.93 | 63.98 | 22.73 | 40.66 |
| Text | TextOp | 82.67 | 43.54 | 20.41 | 31.05 |
| Text | UniAct | 102.44 | 57.95 | 12.76 | 37.05 |
| Text | UH-1 | 96.65 | 48.97 | 18.53 | 38.21 |
| Audio | MotionCraft + GMR | 108.23 | 57.88 | 19.74 | 40.05 |
| Video | Video2Robot + GMR | 119.34 | 91.98 | 23.73 | 34.97 |
| Video | GEM + GMR | 114.88 | 78.49 | 20.98 | 36.57 |
| Trajectory | BFM-Zero | 95.39 | 95.39 | — | — |
| Rhythm | MotionCraft | 96.05 | 96.05 | — | — |

- **v1→v2 更正：** 旧归档中 GEM 文本 BS-L2 13.97、MotionCraft 文本 19.93、Video2Robot 23.55 等不再对应 v2 Table 2；上表以 v2 为准。
- **正文 Q1–Q6：** BG 强并不保证 BS 强；空间补全弱于时间补全（附录 Table B.7）；Body Restrain 在约束中最具挑战（Table B.8 可查）；跨模态组合仍难。TextOp 的 Contact Sliding 在表中明显偏高，**该指标越低越好**，不可把 7.620 解读为优势。
- **汇总口径提醒：** Table 2 多行 BS-L3 数值大于 BS-L2，但层间任务和可用样本不同；不要直接拿两个跨任务均值验算单样本乘积公式，也不要据此断言同一任务上 Level 3 更容易。

## 实验适配与复现限制（附录 D–E）

- 基线按支持模态选择，正文称 9 个 model families；§5.1 的名称（GENMO / PromptHMR）与 Table 2（GEM / Video2Robot）存在不完全一致，核对具体仓库与权重时应采用各实验表与适配协议的名称，避免默认为同名。
- 音频语义条件先经 faster-whisper 转写；文本条件经 Qwen2.5-3B-Instruct 规范化。模型原始输入接口与权重保持不变；输出被转换到统一 G1 / SONIC 表示，人体生成结果使用 GMR，UH-1 将 H1 27 DoF 映射到 G1 29 DoF。
- Task-level Table B.7（条件）与 B.8（约束）已核对；报告数值是作者对可运行有效样本的统计，不等价于所有原始数据实例上的成功率。
- 使用模拟重定向与 SONIC 跟踪验证，不能直接当作全部方法在真实 G1 上完成了同规模硬件验证。

## 局限与许可证

- 主数据偏向 Daily/Locomotion/Sports/Performance（共 83.92%）；单机器人、单主体、以无复杂物理交互动作居多；不覆盖多机器人协调、物体/人接触或长时程规划。
- 合成语音与归一化提示词可能缩小与开放环境自然指令之间的差距。
- 衡量 Speed / Amplitude 的二值条件较宽松；方向容差 60°；现有细分评分无法替代安全性、精确轨迹跟踪或真机鲁棒性测评。
- HF 数据卡注明第三方数据（包括 AMASS/BABEL 派生内容）存在许可约束；商用前分别核实资源的使用及再分发条款。
- 部分数据和评测器权重已发布，网页交互式评测入口可用；**完整官方源码尚无公开运行入口**，不能提供已验证的源码时序图。

## Robotics_Notebooks 映射

- [RoboSteer 详情节点](../../wiki/entities/paper-robosteer-behavior-steerability-benchmark.md)
- [行为基础模型](../../wiki/concepts/behavior-foundation-model.md)
- [前沿论文周报](../../wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md)

## 一手来源

- arXiv v2 HTML：<https://arxiv.org/html/2610.10198v2>
- arXiv 摘要：<https://arxiv.org/abs/2610.10198>
- 官方项目：<https://robosteer.github.io/>
- 官方数据卡：<https://huggingface.co/datasets/YanCORANV/RoboSteer>
