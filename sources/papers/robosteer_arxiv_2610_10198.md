# RoboSteer 论文来源归档

## 元数据

- **题目：** Benchmarking Behavioral Steerability in Behavior Foundation Models
- **基准：** RoboSteer
- **作者：** Minghe Gao（浙大-宇树具身智能联合实验室期间完成）, Zhanxi Yan, Jiahui Liu, Wendong Bu, Xiaoting Chen, Qizhou Wang, Yi Su, Siliang Tang, Jun Xiao, Yueting Zhuang, Tat-Seng Chua, Juncheng Li（通讯）
- **单位：** 浙江大学（ZJU）；宇树科技（Unitree）；新加坡国立大学（NUS）；中南大学（CSU）
- **预印本：** https://arxiv.org/abs/2610.10198
- **HTML：** https://arxiv.org/html/2610.10198v1
- **项目页：** https://robosteer.github.io/（见 [sources/sites/robosteer_project.md](../sites/robosteer_project.md)）
- **数据：** Hugging Face 数据集 https://huggingface.co/datasets/YanCORANV/RoboSteer（见 [sources/repos/robosteer.md](../repos/robosteer.md)）
- **版本：** v1，2026-10-07，cs.RO
- **代码：** 论文称官方适配与评测代码「将发布」；项目页截至 2026-10-12 未列 GitHub。

## 论文要解决的问题

现有行为基础模型（BFM）评测主要看「能不能生成人形行为」，无法判断模型是否准确执行速度、方向、顺序、身体部位约束以及多模态组合要求。论文定义「行为可操控性」（behavioral steerability）并建立分层基准。

## 方法与训练信息（基准设计）

1. **三层结构：** Level 1 条件操控（12 个任务族，630,623 实例，81.51%）——完整条件复现（文本/视频/音频/节奏/旋转到动作）、时间补全（预测、回溯、插值、关键帧）、空间补全（上/下半身到全身、目标到达）。Level 2 约束操控（135,329 实例，17.49%）——速度、幅度、方向、顺序、次数、轨迹、身体约束 7 类。Level 3 组合操控（7,764 实例，1.00%）——文本/音频/视频/关键帧交错的多源条件。
2. **数据规模：** 773,716 实例，20 个任务族；38,522 条源人体动作，各配人体视频与渲染骨架视频（共 77,044 序列，91.58 小时，6,683,715 帧）。动作源自网络规模人体视频，经过滤、字幕（Gemini-3-Flash-Preview 等）、人体动作重建、GMR 物理约束重定向到 Unitree G1、SONIC 跟踪验证；语音条件由 Edge-TTS 合成。
3. **指标：** 传统指标 FID、Diversity、Contact Sliding、MM-Distance、R@1/R@5、MPJPE/g-MPJPE、E_vel、BAS-Gen/BAS-Gap；行为生成分 BG=1000·exp(−0.3·FID−1.6·MM-Distance)（系数为经验选择）；行为操控分 BS_l=BG×∏_{k≤l}IR_k，IR 来自机器人执行反馈。
4. **被评模型：** 9 个 BFM（正文列 GENMO、LoM、MotionCraft、TextOp、UniAct、UH-1、PromptHMR、BFM-Zero、LDA；Table 2 为 GEM+GMR、LoM+GMR、MotionCraft+GMR、TextOp、UniAct、UH-1、Video2Robot+GMR、BFM-Zero、LDA+GMR，两处命名不完全一致）。

## 论文报告的结果（Table 2 摘录）

- **文本条件 BG/BS-L1/BS-L2/BS-L3：** GEM+GMR 114.09/68.38/13.97/42.12；LoM+GMR 104.63/66.27/16.08/39.28；MotionCraft+GMR 114.93/63.98/19.93/40.66；TextOp 82.67/43.54/17.95/31.05（Contact Sliding 7.620 最高）；UniAct 102.44/57.95/10.98/37.05；UH-1 96.65/48.97/17.59/38.21。
- **音频条件：** MotionCraft+GMR 108.23/57.88/16.76/40.05；LoM+GMR 102.51/65.06/15.78/38.01；GEM+GMR 100.56/59.75/12.63/39.93。
- **视频条件：** Video2Robot+GMR BG 119.34、BS-L1 91.98、BS-L2 23.55、BS-L3 34.97；GEM+GMR 114.88/78.49/18.09/36.57。
- **轨迹条件：** BFM-Zero BG=BS-L1=95.39（L2/L3 未报告）。**节奏条件：** LDA+GMR 94.01，MotionCraft 96.05（BS-L1=BG）。
- **作者结论：** 对所有模型 BS 均低于 BG（生成强不等于可操控）；复杂度上升时可操控性下降；空间补全弱于时间补全；Body Restrain 为最难约束；Level 3 同时满足多要求困难，作者归因于以语言为中心的条件表达，并建议多模态交错输入。
- **一致性疑点（读取者备注）：** 多行文本/音频中 BS-L3 高于 BS-L2，与「BS 随要求增加而下降」的叙述不一致；BS-L1 与 BG 完全相等的行意味着 IR₁=1，论文需解释；Level 2/3 的细分表（B.7、B.8）本次未读到。

## 复现与解释边界

- 单机器人（G1）、单人动作；未评测与人/物体的物理交互、多机器人协作与长时程规划；日常/行走/运动/表演占源动作 83.92%，瑜伽、体操偏少；语音为合成。
- 数据集许可：HF 数据卡称团队自有数据与第三方派生材料的再分发许可仍在审查，Level 2 的 Order/Times 使用 AMASS 与 BABEL，二者有非商用且不得再分发条款。
- 项目页称 RoboSteer-Model-Resources 仅有文档和 manifest，「可运行的模型包尚未发布」。
- 本次读取到正文约前 100,000 字符，附录 C–E 与完整 Level 2/3 表未读。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-robosteer-behavior-steerability-benchmark.md
- 概念入口：wiki/concepts/behavior-foundation-model.md
- 汇总页：wiki/overview/frontier-humanoid-quadruped-2026-10-05-10-09.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.10198>
- arXiv HTML 全文：<https://arxiv.org/html/2610.10198v1>
- 项目页：<https://robosteer.github.io/>
