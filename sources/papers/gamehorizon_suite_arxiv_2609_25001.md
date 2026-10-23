# GameHorizon Suite 论文来源归档

## 元数据

- **题目：** GameHorizon Suite: Multi-Horizon Data and Evaluation in Gameplay
- **作者：** Yiran Wang、Xingyilang Yin、Junfu Pu、Guangzhi Wang、Kaifeng Li、Mingyu Ouyang、Huiqiang Sun、Lingen Li、Cheng Cheng、Wangbo Yu、Honghao Chen、Xiaodong Cun、Chi-Man Pun、Zhiguo Cao、Ying Shan
- **机构：** ARC Lab, Tencent IEG；Great Bay University；National University of Singapore；Huazhong University of Science and Technology；The Chinese University of Hong Kong；University of Macau
- **预印本：** https://arxiv.org/abs/2609.25001
- **项目页：** https://gamehorizon-suite.github.io/
- **代码仓库：** https://github.com/TencentARC/GameHorizon （截至 2026-10-23，README 显示注释器、评测代码与数据仍在开源计划中）
- **版本 / 日期：** arXiv:2609.25001，项目页及论文于 2026-09-22 发布

## 核心贡献

GameHorizon 将 AAA 游戏数据、自动标注与评测整合为一套多时间尺度套件。GameHorizon-Data 声称包含 21 款游戏的 5,000 小时录像，由 100 名熟练玩家录制，并对齐视频、玩家动作与多层级指令。GameHorizon-Annotator 从动作感知片段开始，逐步向上合并并标注短时操作（L1）、中期目标（L2）和长期策略（L3）。GameHorizon-Bench 提供离线问答与在线分步任务。

评测覆盖 VLM、通用多模态模型、GUI、代码及游戏智能体等模型家族，共报告 47 个模型。离线轨包含 5,000 道 MCQ（3 个主任务与诊断变体）；在线轨含 20 个长程任务、62 个可验证子任务。作者观察到任务时间跨度增加后，成功率下降；在线结果显示即便短程子任务表现尚可，完整长程任务仍有明显难度，提示规划、目标分解与持续执行仍是薄弱环节。

## 结果解读边界

- “47 个模型”是论文整体套件中的评测范围；项目页当前离线主榜展示 44 个模型，在线轨因成本仅测试 12 个模型。
- 5,000 小时是原始录制规模；论文/项目页分别提供有效时长与各类标注统计，不应把录制总时长等同于所有样本均通过过滤。
- 长程失败可反映任务分解、依赖顺序、记忆和动作执行的综合影响，不能仅凭总成功率归因于单一规划模块。
- 截至入库日，官方仓库 README 仍将注释器、基准代码和数据列为待发布；论文与项目页、公开排行榜已发布。

## 对 wiki 的映射

- 详情节点：[GameHorizon Suite](../../wiki/entities/paper-gamehorizon-suite.md)
- 关联节点：[Player-1](../../wiki/entities/paper-player-1.md)：侧重从示范初始化后，在合成环境中通过强化学习习得游戏控制。
