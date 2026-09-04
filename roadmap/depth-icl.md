# 路线（纵深）：如果目标是 ICL（具身上下文学习）

**摘要**：面向"想让机器人读完一条示范就会做新任务、且不动权重"的纵深路线，从「什么才算真 ICL」的判别边界，到示范表征与动作 token、遥操作示范与人类视频两条数据线，再到零梯度上下文 / 快权重 TTT / 记忆增强的选型与推理预算，最后收在规模涌现与评测协议，按 Stage 0–5 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支，与 [模仿学习纵深](depth-imitation-learning.md)（训练期从示范学）构成"训练期学 / 部署期学"的姊妹路线。

## 路线一览

```mermaid
flowchart LR
  S0["<b>Stage 0</b><br/>判别边界<br/><em>映射选择 · 状态估计 · 映射本身</em>"]
  S1["<b>Stage 1</b><br/>上下文装什么<br/><em>示范表征 · action tokenizer</em>"]
  S2["<b>Stage 2</b><br/>遥操作示范线<br/><em>同坐标系 one-shot · 配对数据</em>"]
  S3["<b>Stage 3</b><br/>人类视频线<br/><em>跨具身 · 无动作标签</em>"]
  S4["<b>Stage 4</b><br/>机制选型与预算<br/><em>零梯度 vs 快权重 vs 记忆</em>"]
  S5["<b>Stage 5</b><br/>涌现与评测<br/><em>规模 · 未见任务 · 协议</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#142a3a,stroke:#e67e22,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是已经能训一个 BC / VLA 策略、开始被"每换一个任务就要重新采数据重训"卡住的人——主战场是操作与移动操作，人形全身也在往这条线上靠
- ICL 解决的是 **部署期适应**：不更新权重，把示范/交互证据塞进上下文，让 **观测→动作映射本身** 变化；它不替代语言语义接地（那是 [VLA 纵深](depth-vla.md)），也不替代前向后果建模（那是 [WAM 纵深](depth-wam.md)）
- **先读 Stage 0 再读论文**：2026 年"上下文"一词同时指 metadata 条件化、历史记忆、快权重 TTT 与真 ICL，不先钉死判据会把四类机制混成一锅
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线是主路线 L5（RL 与模仿学习）之后偏"部署期适应"的进阶方向；起点是 One-Shot Imitation Learning（NeurIPS 2017）这一支"读一条示范就执行"的谱系
- 数据侧强依赖 [遥操作纵深](depth-teleoperation.md)（示范怎么采）与 [模仿学习纵深](depth-imitation-learning.md)（示范怎么训）
- 若目标是"听懂指令的桌面 VLA"，先走完 [VLA 纵深](depth-vla.md) Stage 0–2 再从本路线 Stage 0 切入

---

## Stage 0 判别边界：什么才算真 ICL

**先把"上下文在消解哪一类不确定性"钉成三行表，再读论文，否则 π0.7 的 metadata、MemoryVLA 的历史帧与 RoboTTT 的快权重会被一起标成 ICL。**

### 前置知识
- Python + PyTorch 熟练，训过一个行为克隆或 VLA 策略
- [VLA 纵深](depth-vla.md) Stage 0–1 水平（知道 \(a_t=\pi(o_t,\ell)\) 与 action chunking）

### 核心问题
- 标准 VLA 的 **马尔可夫假设** 在部署时怎么破：同一画面多种合法动作、相机/末端变化改写像素–运动学关系、多阶段任务当前帧看不到进度
- 三类不确定性各自被什么上下文消解：**映射选择**（语言 / 目标图 / metadata）、**状态估计**（历史帧 / 记忆 token）、**映射本身**（示范 / 人视频 / 系统辨识片段）——只有第三类是真 ICL
- 判别口诀：读完 **任务示范** 后"怎么做"变了 → ICL；读完 **历史帧** 只是知道"做到哪一步了" → 记忆；读完 **metadata** 只是换执行风格 → 条件化选择
- 通用 LLM 控机器人的"长上下文"多来自 **短时程重试**，不是从示范归纳新映射

### 推荐做什么
- 把手上收藏的 8–10 篇"in-context"论文逐一填进三类不确定性表，标注读完之后到底什么变了
- 对自己的场景写一句话：我缺的是映射、状态还是选择？——缺状态就别去堆示范

### 推荐读什么
- [机器人 In-Context Learning（概念 taxonomy）](../wiki/concepts/robot-in-context-learning.md)（本仓库）— 概念枢纽，三类不确定性与判别口诀
- [π0.7](../wiki/methods/pi07-policy.md)（本仓库）— 映射选择的正面样本：metadata 选映射，不改函数形式
- [LLM 控制接口](../wiki/concepts/llm-robotics-control-interfaces.md) · [Embody](../wiki/entities/anthropic-embody.md)（本仓库）— 通用 LLM 控机器人的代际优势来自短时程重试，截掉远期上下文多数模型不掉分
- [具身大模型家族分类学闭环](../wiki/queries/embodied-fm-taxonomy-loop.md)（本仓库）— ICL 是 VLA 层的"部署期适应旋钮"，不是新的一层

### 学完输出什么
- 一张三类不确定性对照表（上下文内容 × 读完之后什么变了 × 是否真 ICL）
- 拿到新论文能一句话归格，并说清它的代价是"推理变长"还是"要跑梯度"

---

## Stage 1 上下文里装什么：示范表征与动作 token

**同一条示范可以是 token 序列、图节点、关键点或一段结构化文字；抽象度直接决定"好归纳"还是"信息全"。**

### 前置知识
- Stage 0 内容
- Transformer 序列建模 + 扩散 / 流匹配动作头的使用级直觉

### 核心问题
- 五类示范表征各自的代价：图像/状态/动作 **token 交错序列**（ICRT）、**图 diffusion**（Instant Policy）、**关键点 + 文本**（KAT）、**示范 embedding + cross-attention**（BPP）、**结构化计划 + 子目标 + 2D/3D 运动 verbalization**（StellaVLA）
- **抽象度权衡**：抽象高则归纳容易但丢接触与力信息；抽象低则保留全信息但人–机对应关系难建立
- **Action tokenizer 的平滑性**（如 LipVQ-VAE）为什么直接决定归纳出的控制能不能执行：相邻动作在 latent 空间跳变，示范归纳出来也是抖的

### 推荐做什么
- 选两种表征（如 token 序列 vs 结构化 verbalization）在同一批示范上各做一次编码，对比上下文长度与信息损失
- 给自己的任务列一张"必须保留 / 可以压成一个 token"的信息清单——接触力通常在第一栏

### 推荐读什么
- [StellaVLA](../wiki/entities/paper-stellavla-structured-icl-vla.md)（本仓库）— 离线 VLM 把原始示范转成结构化 in-context 示范；VLA-Arena overall 0.63、LIBERO 98.8%、LIBERO-Plus 85.1%；截至入库日无可运行官方代码
- [BPP（Behavior Prompting Policy）](../wiki/entities/paper-behavior-prompting-policy.md)（本仓库）— 单次人类示范作 behavior prompt + cross-attention；训练与部署代码已开源
- [机器人 ICL 综述归档（具身智能之心，2026-08-25）](../sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md)（本仓库）— ICRT / Instant Policy / KAT / SynthICL / RICL 的表征谱系
- [接触力旋量闭环](../wiki/queries/contact-wrench-closed-loop.md)（本仓库）— 高抽象表征把力与接触细节丢在哪里

### 学完输出什么
- 一份"示范表征选型便签"：上下文长度、归纳难度、丢失信息三栏
- 能解释为什么自己的 one-shot 结果"看着像但插不进去"——通常是抽象度或 action tokenizer 的问题

---

## Stage 2 遥操作示范线：同坐标系的 one-shot 归纳

**最干净的一条线：示范与执行同本体同坐标系，难点从"对齐"退化成"训练目标与配对数据"。**

### 前置知识
- Stage 1 内容
- 会用遥操作或脚本采一批同任务多条示范（见 [遥操作纵深](depth-teleoperation.md)）

### 核心问题
- 训练目标怎么构造："一条示范 + 一次查询"，优化的是 **读完示范后的执行表现**，而不是单纯拟合动作
- 配对数据从哪来：同任务多条示范互为 prompt/query；仿真程序化生成（SynthICL）
- **退化风险**：上下文里最近的 chunk 与当前动作高度相关，模型会学成"复制最近 chunk"——Qwen-RobotManip 用 **stochastic context sampling** 压这条捷径
- 已有预训练 VLA 怎么后装 ICL：RICL 类在 π0-FAST 上做小规模 in-context post-training，不必从头训

### 推荐做什么
- 用同任务 20–50 条示范做一次 in-context post-training，评测必须留出 **未见物体 / 未见摆位**
- 做一次"复制捷径"消融：把上下文示范换成同任务的另一条、无关任务的一条、以及空上下文，三档成功率若无差异说明模型没在读示范

### 推荐读什么
- [Imitation Learning](../wiki/methods/imitation-learning.md)（本仓库）— one-shot / few-shot 训练目标的传统路线
- [Qwen-RobotManip](../wiki/entities/qwen-robot-manip.md)（本仓库）— in-context 适配 + stochastic context sampling；>38,100 h 开源预训练
- [Foundation Policy](../wiki/concepts/foundation-policy.md)（本仓库）— ICL 是部署期适应手段，不改变"大规模预训练通用策略"母类定义
- [RealAB 14 篇技术地图](../wiki/overview/realab-14-papers-technology-map-2026.md)（本仓库）— BPP 等 in-context 操作索引

### 学完输出什么
- 一条可复现的 in-context 训练配方（配对构造 + 上下文采样策略 + 未见集划分）
- 一张"空上下文 / 错上下文 / 对上下文"三档消融表——这是 ICL 声明的最低证据门槛

---

## Stage 3 人类视频线：跨具身、无动作标签

**人视频便宜且多样，但没有动作标签、还隔着 embodiment gap；这条线的全部工程量都在"怎么把视频变成可归纳的示范"。**

### 前置知识
- Stage 2 内容
- 了解动作重定向与人–机对应（[动作重定向纵深](depth-motion-retargeting.md) Stage 0–1 水平）

### 核心问题
- 无动作标签怎么办：视频–轨迹配对 + 对比对齐（Vid2Robot）、腕部位姿重定向（MimicDroid）、语义关键点统一人机观测（Point Policy）
- 怎么防"学到人类特有线索"：MimicDroid 用 **随机块遮挡（patch masking）** 降低对人手外观的过拟合
- 怎么防"捷径"：Zero-WAM 用 **in-context future chunk prediction（IFP）** 逼模型真读人视频规格，而不是从当前帧猜
- 进度对齐：单条真人视频要落到"做到哪一步"，HOST 用 **进度流形 + 自接地级联**；不改权重、平均 29 秒完成一次适配
- **能力边界**：意图级模仿到 L3（功能替代）会崩——Imitator Game 报告未见任务零样本 <13%

### 推荐做什么
- 用 3–5 条自拍人视频跑一次开源 one-shot 管线（HOST 代码与 HF 权重已开），记录从"拿到视频"到"机器人动"的墙钟时间
- 把失败样本按 **对齐失败 / 重定向失败 / 归纳失败** 分三类计数——这三类的修法完全不同

### 推荐读什么
- [HOST](../wiki/entities/paper-host-one-shot-human-video.md)（本仓库）— 单条人视频、平均 29 s、不改权重；八任务 62%，已掌握技能保留约 99%；代码 + HF 权重已开源
- [MimicDroid](../wiki/entities/paper-notebook-mimicdroid-in-context-learning-for-humanoid-robo.md)（本仓库）— 仅用人类玩耍视频训练人形 ICL：轨迹配对 + 运动学重定向 + patch masking；配套开源仿真基准，真机成功率近两倍
- [Zero-WAM](../wiki/entities/paper-zero-wam.md)（本仓库）— 人视频作 in-context 任务规格；HumanGen 74.2K 配对 / 8.6K 任务；RoboTwin 2.0 未见 46.95%
- [The Imitator Game](../wiki/entities/paper-imitator-game.md)（本仓库）— L0–L3 意图级模仿基准；L3 功能替代崩溃，给这条线画能力上界
- [跨具身知识链](../wiki/overview/hub-cross-embodiment.md)（本仓库）— 人视频 / 仿真 prompt→真机与重定向、域随机的机制差异

### 学完输出什么
- 一条能跑通的"人视频 → 机器人执行"最短链路，并说得清哪一环最脆
- 一份人视频 ICL 的能力边界记录：什么样的任务变体它归纳得了，什么样的归纳不了

---

## Stage 4 机制选型与推理预算：零梯度上下文 vs 快权重 TTT vs 记忆增强

**三种机制目标重叠、代价完全不同；选错的代价通常不是精度，而是控制回路的每步预算。**

### 前置知识
- Stage 3 内容
- 对真机部署延迟、异步 action chunk 有直觉（[VLA 纵深](depth-vla.md) Stage 4 水平更佳）

### 核心问题
- **零梯度上下文**：不动权重、可回退，但上下文变长直接抬高 **每步推理成本**——控制回路要高频出动作，这不是语言模型那种"首 token 延迟"问题
- **快权重 TTT**：RoboTTT 在 GR00T N1.7 内嵌 TTT 层，把上下文压进固定大小 fast weights，扩到 8K 步（约 5 min）且推理延迟不随上下文增长；代价是要跑梯度、难回退、在线安全要求更高
- **同人视频、不同机制的直接对照**：WAM-TTT 在 New 家庭场景 9 任务上 46.2% avg progress，显著优于 WAM-ICL 的 7.1%（同论文自报）——这条数字是"纯上下文不够用"的最强单点证据，但只在该 setting 成立
- **记忆增强不是 ICL**：KEMO / EventVLA / Chronos / GMP 解决的是部分可观测（做到哪一步了），读完不改变观测→动作函数
- 跨篇数字不可直接比较：四条路线互知却无同 backbone 并排实验，选型应按 **漂移轴**（场景 / 构型 / 扰动 / 任务 / 时长）而不是排名

### 推荐做什么
- 用 [四路线对比页](../wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md) 的坐标系给自己的场景定位：漂移发生在哪根轴上，就选哪一族
- 算一笔账：上下文长度 × 每步推理耗时 vs 控制频率要求，先确定预算再挑方法

### 推荐读什么
- [WAM-TTT × RoboTTT × StellaVLA × Zero-WAM 四路线对比](../wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md)（本仓库）— 快权重 vs 零梯度的定性坐标系（不做性能排序）
- [RoboTTT](../wiki/entities/paper-robottt-test-time-training-vla-context.md)（本仓库）— fast weights 把上下文扩到 8K 步，长程双臂装配报告约 +87%
- [WAM-TTT](../wiki/entities/paper-wam-ttt-human-video-test-time-steering.md)（本仓库）— 冻结 WAM 上的人视频测试时 steering；46.2% vs WAM-ICL 7.1%
- [VLA](../wiki/methods/vla.md) 长程记忆增强小节（本仓库）— [KEMO](../wiki/entities/paper-kemo-event-driven-keyframe-memory-vla.md) / [EventVLA](../wiki/entities/paper-eventvla-visual-evidence-memory.md) / [GMP](../wiki/entities/paper-gated-memory-policy.md) 的记忆路线，对照读
- [WAM 纵深](depth-wam.md) Stage 4（本仓库）— 部署期"执行 / 修正 / 筛选"接口，与本阶段选型同构

### 学完输出什么
- 一张"我的栈该用零梯度上下文 / 快权重 TTT / 记忆增强哪一类"的选型表，含每步预算列
- 能说清为什么自己不选另外两类——理由要落在漂移轴与预算上，不是榜单名次

---

## Stage 5 规模涌现、评测与产业对照

**产业侧宣称"涌现式 one-shot"，学术侧证明"可负担规模下需要显式机制"——两边都对，前提是别把闭源自报数字当作可复现结论。**

### 前置知识
- Stage 4 内容

### 核心问题
- **两条相反路线**：GEN-1.5 报告 **无显式 ICL 训练** 下涌现 physical prompting（3–12 s 示范，one-shot 约 59%，10 步微调约 83%）；S1 把 ICL 写成 **预训练目标本身**（任务只经视频示范指定），宣称未见任务最长约 10 分钟、100k h 档未见 66% vs 语言 VLA 9%——两者均为 **闭源自报**，需独立验证
- **涌现条件**：世界模型侧的研究把"上下文与多样性"作为 ICL 涌现的前提变量；何种数据分布 / 规模可预测涌现仍是开放问题
- **评测协议**：未见集怎么划（未见物体 / 未见场景 / 未见任务 / 未见时长）、one-shot 与 few-shot 怎么报、示范质量怎么控——跨篇数字不可直接比较
- **长上下文 scaling**：何信息必须逐帧保留、何信息可压成一个 token，仍无定论；这直接决定 ICL 能不能进高频控制回路

### 推荐做什么
- 给自己的评测写一份协议：未见集定义、示范条数与时长、随机种子、失败模式分类——这是本方向最容易被质疑的地方
- 把 GEN-1.5 / S1 的宣称逐条标注为"可复现 / 闭源自报 / 假设性解释"，训练自己的证据分级习惯

### 推荐读什么
- [GEN-1.5](../wiki/entities/generalist-gen15-one-shot.md)（本仓库）— 涌现式 one-shot 的产业样本（闭源自报）
- [S1（Skild）](../wiki/entities/skild-s1.md)（本仓库）— 显式 ICL 预训练；未见长程操作（闭源自报）
- [ICL Emergence: Context and Diversity Matter](../wiki/entities/paper-sa-2509-22353-icl-emergence-context-and-diversity-matter-the-e.md)（本仓库）— 世界模型侧的涌现条件研究
- [具身缩放律](../wiki/concepts/embodied-scaling-laws.md) · [数据飞轮](../wiki/concepts/data-flywheel.md)（本仓库）— 规模与数据循环侧的背景
- [具身模型测评纵深](depth-embodied-eval.md)（本仓库）— 未见集划分与成功率之外的过程指标

### 学完输出什么
- 一份自己的 ICL 评测协议文档，能直接贴进论文或技术报告
- 一句话答辩：我的方法在哪根漂移轴上、用什么机制、代价是多少、证据等级是什么

---

## 快速入口汇总

| 阶段 | 核心问题 | 本仓库入口 |
|------|---------|-----------|
| Stage 0 | 什么才算真 ICL | [机器人 In-Context Learning](../wiki/concepts/robot-in-context-learning.md) |
| Stage 1 | 示范表征与 action token | [StellaVLA](../wiki/entities/paper-stellavla-structured-icl-vla.md) |
| Stage 2 | 遥操作示范线与配对数据 | [Qwen-RobotManip](../wiki/entities/qwen-robot-manip.md) |
| Stage 3 | 人类视频线与跨具身 | [HOST](../wiki/entities/paper-host-one-shot-human-video.md) |
| Stage 4 | 零梯度 / 快权重 / 记忆选型 | [四路线对比](../wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md) |
| Stage 5 | 涌现、评测与产业对照 | [GEN-1.5](../wiki/entities/generalist-gen15-one-shot.md) · [S1](../wiki/entities/skild-s1.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 其它纵深路径：
  - [遥操作（人形全身遥操作 + 手指遥操作 → 示范数据/实时接管）](depth-teleoperation.md) — 上游：ICL 的示范从哪来
  - [模仿学习与技能迁移](depth-imitation-learning.md) — 姊妹路线：训练期从示范学 vs 部署期从上下文学
  - [VLA（视觉-语言-动作模型）](depth-vla.md) — ICL 是 VLA 层的部署期适应旋钮
  - [WAM（世界–动作模型）](depth-wam.md) — Stage 4 的部署职责三分与本路线选型同构
  - [BFM（人形行为基础模型）](depth-bfm.md) — 身体级协调；可与上下文适应分层叠用
  - [具身模型测评（认知 → 世界模型保真 → 策略成功率 → sim↔real 校准）](depth-embodied-eval.md) — Stage 5 未见集与协议的展开版
  - [接触丰富的操作任务](depth-contact-manipulation.md) — Stage 1 高抽象示范丢掉的力/接触信息在这里补
  - [动作重定向（人体动作 → 机器人参考轨迹）](depth-motion-retargeting.md) — Stage 3 人–机对应的工程侧
  - [Loco-Manipulation（移动操作）](depth-loco-manipulation.md)
  - [导航（SLAM → Nav2 → VLN → 导航 VLA）](depth-navigation.md)
  - [动作生成（文本/多模态 → 人形动作）](depth-motion-generation.md)
  - [人形 RL 运动控制](depth-rl-locomotion.md)
  - [力矩控制电机设计（指标 → 电磁热 → FOC 力矩闭环）](depth-torque-motor-design.md)
  - [传统模型控制（LIP/ZMP → MPC → WBC）](depth-classical-control.md)
  - [人形整机硬件设计（指标预算 → 机械 → 电气 → 通信 → 整机验收）](depth-humanoid-hardware-design.md)
  - [安全控制（CLF/CBF）](depth-safe-control.md)
  - [感知越障（Perceptive Locomotion）](depth-perceptive-locomotion.md)
  - [人形足球（全向行走 → 感知踢球 → 多机战术）](depth-humanoid-soccer.md)
  - [人形群控展演（群舞同步 → 编队走位 → 群体特技）](depth-humanoid-swarm-performance.md)
  - [人形拳击（动作跟踪 → 潜空间技能 → 对抗自博弈）](depth-humanoid-boxing.md)
  - [Sim2Real（域差画像 → 执行器对齐 → 鲁棒训练 → 真机部署）](depth-sim2real.md)
  - [Real2Sim（真实世界 → 可仿真资产/场景/孪生）](depth-real2sim.md)
- 人形控制全景图：[Humanoid Control Roadmap](../wiki/roadmaps/humanoid-control-roadmap.md)
- 技术栈地图：[tech-map/dependency-graph.md](../tech-map/dependency-graph.md)

## 参考来源

本路线基于以下原始资料与 wiki 编译页的归纳：

- [机器人 In-Context Learning 概念页](../wiki/concepts/robot-in-context-learning.md)
- [WAM-TTT × RoboTTT × StellaVLA × Zero-WAM 四路线对比](../wiki/comparisons/wam-ttt-robottt-stellavla-zero-wam-embodied-icl.md)
- [sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md](../sources/blogs/wechat_embodied_heart_robot_icl_gen15_survey_2026-08-25.md) — ICL taxonomy 综述（含 One-Shot Imitation Learning, NeurIPS 2017 起点谱系）
- [sources/blogs/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31.md](../sources/blogs/wechat_meiri_zhineng_embodied_icl_four_papers_2026-08-31.md) — 四篇可核对论文纵横向解读
- [sources/blogs/generalist_gen15_one_shot.md](../sources/blogs/generalist_gen15_one_shot.md) · [sources/blogs/skild_s1_in_context_learning.md](../sources/blogs/skild_s1_in_context_learning.md) — 产业侧 one-shot 宣称（闭源自报）
