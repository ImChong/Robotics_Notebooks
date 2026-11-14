# 2026 BEHAVIOR Challenge｜2026 家庭具身智能挑战赛官方资料

- **类型：** Stanford Vision and Learning Lab 主办的 CoRL 2026 benchmark challenge
- **英文标题：** 2026 BEHAVIOR Challenge
- **中文标题：** 2026 BEHAVIOR 家庭机器人挑战赛
- **项目主页：** https://behavior.stanford.edu/challenge/index.html
- **数据与格式：** https://behavior.stanford.edu/challenge/dataset.html
- **评测原文：** https://behavior.stanford.edu/challenge/evaluation.html
- **训练基线：** https://behavior.stanford.edu/challenge/baselines.html
- **提交说明：** https://behavior.stanford.edu/challenge/submission.html
- **任务演示：** https://behavior.stanford.edu/challenge/tasks/index.html
- **排行榜：** https://huggingface.co/spaces/behavior-1k/2026-challenge-leaderboard
- **代码：** https://github.com/StanfordVL/BEHAVIOR-1K
- **数据集：** https://huggingface.co/datasets/behavior-1k/2026-challenge-demos
- **核查日期：** 2026-11-14
- **核对范围：** 通读官方 Challenge 首页、Dataset 数据格式与统计、Baselines 训练/评测步骤、Evaluation and Rules 的单赛道、部署、评测命令、时间预算、Q-score、实例划分与效率指标；结合 BEHAVIOR-1K main 上评测入口与核心组件。冠军方法仅作相关工作交叉引用，不当作赛制本身。
- **通读：** 已完成

## 赛事定位

CoRL 2026（2026-11-09 至 11-12，奥斯汀）期间展示的第二届 BEHAVIOR Challenge，含 **100 个完整家务任务、7 个场景（4 个新场景）**。挑战结合室内跨房间导航、长程决策、双臂精细操作及物体状态变化。2026-07-02 启动，2026-10-16 截止提交，官方页面列获奖公布日为 2026-11-04。

## 输入、演示与基线

- 单一 challenge track，**评测时仅允许 RGB、深度与机器人本体感知**，不可使用仿真真值分割、物体状态/真值位姿、全局机器人位置等特权观测；训练时可使用特权信息。
- 20,000 条 JoyLo 人类遥操作示范，总计约 1,950 小时，覆盖 100 任务，LeRobot v3 格式，含视频、proprio、动作和技能/子任务标注；官网基线页称完整数据约 3.27 TB。
- 基线为 π0.5（挑战赛 OpenPI fork）和 GR00T N1.7（挑战赛 Isaac-GR00T fork）。参赛者可选 R1Pro 或 OmniGibson 支持的自定义机器人，也可使用 IL、RL、TAMP、SLAM 或 LLM 辅助方法，前提是不越过评测输入限制。

## 评测指标与协议

- **主指标 Q-score**：每条 rollout 在结束时满足的 BDDL goal predicates 数量 / 该任务总目标谓词数；对 100 任务取均值排名，**允许部分达成**。
- **完整任务成功率**：全部目标满足的比例；与 Q-score 不同，不能混称。
- **次指标**：仿真耗时、底盘导航距离、末端执行器位移；按每任务 200 条人类演示归一化，主分相同才用于裁决。
- 每任务 200 条训练演示。报告用公开实例索引 0–9，每实例跑一次；公开索引 10–19 可自测；决赛另有隐藏实例。任务默认超时为人类平均示范长度的 1.5 倍，细节见官方评测页。
- 评测用 websocket 策略服务，必须产出每次 rollout 的 JSON 与视频；视频包含头部及腕部相机。多环境评测允许每 rollout 独立 policy endpoint；action-chunk replay 可选且默认关闭。

## 评测源码与版本注意

- 官方文档要求使用 BEHAVIOR-1K 的 **v3.9.3** tag，而非 v3.9.0；示例入口：python -m omnigibson.eval.eval。
- 2026-11-14 通过 GitHub 连接读取 **main** 的 OmniGibson/omnigibson/eval/eval.py（blob f937e00b）、evaluator.py（blob 8404a659）、eval/utils/network_utils.py（blob 06fcb45c）和 eval/utils/score_utils.py（blob 65cfc123）；入口参数、BatchedEvaluator、WebsocketPolicy、计分数据与官网描述可互相对应。
- **尚未验证 v3.9.3 的代码快照**：通过 GitHub 文件接口使用 ref=v3.9.3 时返回“no commit found for the ref”。这仅反映本次接口读取结果，不等于断言 tag 不存在；本次未实际运行 OmniGibson、策略服务器或提交评测。
- 开源边界：赛事评测环境、示例管线与数据集公开；大型资产与数据仍需自行下载并遵守许可。**赛事评测代码不等于冠军 HiPhys-RSI 的开源方案**。

## 对 Wiki 的映射

- [2026 BEHAVIOR Challenge 独立实体](../../wiki/entities/behavior-challenge-2026.md)
- [BEHAVIOR-1K 原始基准](../../wiki/entities/behavior-1k.md)
- [HiPhys-RSI 冠军技术报告](../../wiki/entities/paper-hiphys-rsi.md)
- [源码入口归档](../repos/behavior-1k-challenge.md)
