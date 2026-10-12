# AgentGarten: Code Worlds for Evolving Agents（arXiv:2610.12374）

> 来源归档（paper / technical report）

- **标题：** AgentGarten: Code Worlds for Evolving Agents
- **arXiv：** <https://arxiv.org/abs/2610.12374>（[HTML](https://arxiv.org/html/2610.12374v1)、[PDF](https://arxiv.org/pdf/2610.12374)）
- **项目页：** <https://mirros-lab.github.io/agent-garten/>
- **代码：** <https://github.com/MirroS-Lab/AgentGarten>
- **作者：** Jiawei Chi, Shangchen Miao, Zhiyuan Shi, Kailu Wu, Hanyang Wang, Weiliang Chen, Qiyu Dai, Jinshan Ren, Jun Gao, Mingsheng Long, Yueqi Duan, Jiangran Lyu, Jialong Wu, Fangfu Liu
- **机构：** MirroS、清华大学、北京大学（按论文署名）
- **入库日期：** 2026-10-12
- **一句话说明：** 用显式仿真器和游戏引擎维护 code world 的状态与规则，共享神经视频渲染器把深度/法线等几何条件变成智能体视觉；预训练智能体在每轮后把经验写成 playbook 传给下一轮。
- **对 wiki 的映射：** [`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)

## 核心论文摘录

### 1. 系统边界：引擎维护世界，神经渲染器生成观察

- **出处：** Abstract、§2、Figure 1。
- **要点：** 世界程序在 simulator/game engine 中保存持久状态并执行程序定义的交互规则；统一接口导出深度图或表面法线等结构化条件。共享渲染器结合条件、外观参考和历史生成视觉观察。渲染器不是状态转移模型，也不代替仿真器决定物体如何移动。
- **对 wiki 的映射：** AgentGarten 实体页「核心架构」与流程图；连接 [Code-as-World](../../wiki/entities/paper-code-as-world.md) 说明同团队的可执行世界表示方向。

### 2. 低步数流式视频渲染

- **出处：** §3–4，尤其 §3.5、§4.3、Table 2。
- **要点：** 从 Cosmos3-Nano 适配几何条件视频生成：先做首帧条件下的双向 rectified-flow 训练，再做带 KV cache 的块因果自回归训练，最后用 Adversarial Forcing 蒸馏少步 student。精确 replay 让后续块损失可回传到历史重新编码；真实数据判别器与精确 R1/R2 正则提升视觉质量。
- **吞吐：** 480×832 视频在单张 NVIDIA H100 上报告 36.5 FPS；稳态每块 16 帧总 wall time 438.8 ms，含条件编码、4 步去噪、tiny decoder 与 host transfer。采用手写 kernel、CUDA graph 和 FP16 tiny decoder。这是暖机后 renderer throughput，不含环境 action 执行和智能体推理时延。
- **对 wiki 的映射：** 实体页「渲染器训练」「工程实践」和运行时序图。

### 3. Hide-and-seek：跨轮技能沉淀

- **出处：** §5.2，Figures 8–9、Table 3；配套项目博客解释实验协议。
- **要点：** hider/seeker 起始 playbook 为空；每轮包含 5 局，智能体通过第一人称渲染帧观察，不接收物体坐标、隐藏状态或对手私有观察。它们提交短 Python 动作程序；每轮后复盘允许使用的行动记录和视觉证据，将技能/失败分析写入文本 playbook，由下一轮的新智能体继承。
- **结果：** hider 在第 4 轮用面板遮挡入口形成 shelter；seeker 在第 10 轮把坡道搬到墙边并越墙。属于少量演示轮次的行为里程碑。
- **重要比较边界：** OpenAI 2019 自博弈 PPO 研究报告约 2500 万局后出现 shelter、约 1 亿局后出现 ramp 使用。AgentGarten 与该工作在初始化、预训练先验、观察空间、角色/游戏协议及优化方法上均不同；round 也不是等价于一次 RL episode。因此这组数用于说明“预训练智能体可用少量复盘把已有工具知识落地”，不能解读成同设置下的样本效率比或“从零学习物理”的倍数提升。

### 4. 多个 code worlds 的演示

- **出处：** §5、Figure 7、Table 4。
- **要点：** 同一套 renderer 接口还演示 companion dog、one-lane bridge、sheep herding、quarry loader 四个世界；各自采用任务相关动作、时间限制和终止分数。每个展示运行四轮。不同世界度量不相同，论文没有据此给出跨任务统计显著性或大规模泛化保证。
- **对 wiki 的映射：** 实体页「实验与结果」。

## 复现与开源核查

- **官方仓库现状（2026-10-12）：** Apache-2.0 仓库开放神经渲染器的训练配方、checkpoint 与块式流式推理；README 依赖 Python 3.11+、PyTorch 2.9+ 和 CUDA，并使用 Cosmos3-Nano / Wan 2.2 VAE 权重。
- **仍待发布：** README TODO 将实时渲染 server / interactive frontend，以及 code worlds 与 agent practice loop 标为未完成。因此公开 repo 目前不足以直接复跑完整的 hide-and-seek/playbook 系统实验。
- **性能复现条件：** README 的 `Cosmos3Stream` + `prepare_serving` 路径需先做 kernel JIT 与 CUDA graph capture；论文 36.5 FPS 为固定 H100 / 480×832 / 稳态 16 帧块条件下的渲染器测量。
- **对 wiki 的映射：** 实体页「开源状态与工程实践」「局限与风险」。

## 对 wiki 的映射

- 沉淀统一实体：[`wiki/entities/paper-agentgarten.md`](../../wiki/entities/paper-agentgarten.md)
- 项目页归档：[`sources/sites/agentgarten-project.md`](../sites/agentgarten-project.md)
- 官方博客归档：[`sources/blogs/mirros-agentgarten.md`](../blogs/mirros-agentgarten.md)
- 官方仓库归档：[`sources/repos/agentgarten.md`](../repos/agentgarten.md)
