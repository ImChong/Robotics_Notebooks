# RoboICL: Embodied In-Context Learning with GPT-6 Astra

> 一手论文归档（paper）

- **标题：** RoboICL: Embodied In-Context Learning with GPT-6 Astra
- **中文标题：** RoboICL：利用 GPT-6 Astra 实现具身上下文学习
- **类型：** paper / embodied-icl / test-time-adaptation / multimodal-action
- **arXiv：** https://arxiv.org/abs/2609.34261
- **HTML：** https://arxiv.org/html/2609.34261v1
- **PDF：** https://arxiv.org/pdf/2609.34261
- **版本：** arXiv v1（2026-09）
- **作者：** Fangcheng Liu、Yeqing Shen、Anda Cheng 等（以 v1 原文署名为准）
- **官方项目：** https://mosi-ai.github.io/RoboICL-GPT6-Astra.github.io/
- **官方代码：** https://github.com/Mosi-AI/RoboICL
- **入库日期：** 2026-11-21
- **核对范围：** arXiv v1 HTML 的方法、任务设置、RoboDojo 与真机实验、消融、失败案例、限制及附录的评测协议；官方仓 README、roboicl/run.py、roboicl/rollout.py、roboicl/policy_server.py 与 policy/ 模块索引。
- **通读：** 已完成

## 方法原文摘要

1. 冻结通用多模态模型 GPT-6 Astra，不进行机器人专用权重更新；模型根据当前三机位观测与本体状态，读取示范上下文（TRAIN）与执行上下文（LIVE）。
2. TRAIN 的示范记录以及 LIVE 的交互记录统一为观测→动作块→执行回执→结果观测；显式回执防止把计划动作误当作真实完成动作。
3. 有界锚点记忆保留最早、时间锚点、最新完整交互；当历史被压缩时标明轨迹间隔，避免模型错误假设两段相邻。
4. 控制接口是双臂笛卡尔动作增量与夹爪开合（14 维），执行一个动作块后获取真实反馈再修正；任务自适应 horizon 在官方配置中记录。
5. 官方支持零样本 J=0/B=25 和一示范 J=12/B=12 两种配置；一示范每个窗口保存动作块与前后观测，而非简单截取 12 张视频帧。

## 实验和证据分级

- RoboDojo 30 任务基准：作者报告 RoboICL 归一化平均进度 50.64，最强对照 33.68；Open 类任务为 zero-shot，其余任务 one-shot，必须按论文协议理解。
- 作者报告 3 项真实机器人任务：zero-/one-/three-shot 进度 14.45 / 63.33 / 78.89；样本任务有限，不是整个真机分布的统计结论。
- Jev 门控是在开发任务集上减少 33–48% 的 API 调用；不可解释为所有任务的普适成功率增益。
- 论文记录推理请求与上下文成本；API 调用延迟不能当作可直接跑高频电机内环的证据。

## 原文局限

- 依赖大模型调用、长上下文和稳定网络，动作块低频闭环必须与机器人驱动层解耦。
- RoboDojo 任务集和真实机器人任务数量有限；新物体/织物等未见分布仍可能失败。
- 论文指标、对照、真机结果应标注各自协议，避免简单横比。

## 源码对应

- 运行入口：```python -m roboicl.run```（Python 代码围栏标识为文字，具体命令见 README）。
- 运行时序：roboicl/run.py 启动/校验环境与策略服务；roboicl/policy_server.py 通过 XPolicyLab WebSocket 调用策略；roboicl/rollout.py 在 env.get_obs、model_client.get_action、env.take_action 与 update_obs 之间闭环。
- upstream.lock.json、data.lock.json 和 one_shot_j12_b12.json 固定依赖、资产和示范选择，不以 README 的安装成功替代真机复现。

## 对 wiki 的映射

- [RoboICL 唯一论文节点](../../wiki/entities/paper-roboicl.md)
- [具身 ICL 路线](../../roadmap/depth-icl.md)
