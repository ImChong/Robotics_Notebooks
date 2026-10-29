# 路线（纵深）：如果目标是跨本体迁移（Cross-Embodiment）

**摘要**：面向"想让一个策略适配不同本体——换夹爪、换相机、换一台人形甚至换一类形态都不必从零重训"的纵深路线，从「本体差异到底差在哪几层」的画像，到统一动作/状态接口、形态感知架构、多本体混训与数据配比，再到人→机这条最大跨度的迁移，最后收在新本体的低成本适配与验收，按 Stage 0–5 串通核心方法；本路线是 [运动控制主路线](motion-control.md) 的一条分支，承接 [动作重定向纵深](depth-motion-retargeting.md)（参考层的跨骨架映射）向"策略层跨本体"延伸。站内也写作"跨具身迁移"，两词同义。

## 路线一览

```mermaid
flowchart LR
  S0["<b>Stage 0</b><br/>本体差异画像<br/><em>五层差异 · 迁移场景谱</em>"]
  S1["<b>Stage 1</b><br/>统一接口<br/><em>动作/状态空间 · 槽位 · 潜动作</em>"]
  S2["<b>Stage 2</b><br/>形态感知架构<br/><em>共享骨干 + 本体头 · 图/序列编码</em>"]
  S3["<b>Stage 3</b><br/>多本体混训<br/><em>正迁移条件 · 随机形态 · 配比</em>"]
  S4["<b>Stage 4</b><br/>人→机迁移<br/><em>重定向 · 本体无关表征</em>"]
  S5["<b>Stage 5</b><br/>新本体适配与验收<br/><em>LoRA · 上下文适应 · 负迁移</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#142a3a,stroke:#e67e22,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是已经能在 **一台** 机器人上训出可用策略（WBT / RL locomotion / BC / VLA 任一），开始被"每换一台机器就要重采数据、重训一遍"卡住的人
- 跨本体迁移解决的是 **策略层复用**：同一份权重（或同一个共享骨干）服务多个本体；它不等于 [动作重定向](depth-motion-retargeting.md)（只映射参考轨迹），也不等于 [Sim2Real](depth-sim2real.md)（同一本体的仿真↔真机）——但三者常串联
- **先读 Stage 0 再选方法**：同样叫"跨本体"，换夹爪、换同族人形、人体→四足是三类完全不同的问题，选错层级的方法比不迁移更糟
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线是主路线 L5.4（动作重定向）之后偏"一个策略服务多本体"的进阶方向，操作侧与 L10（VLA / Foundation Policy）交汇；起点是 Devin 等把策略拆成 **任务模块 × 机器人模块** 的模块化多机器人迁移（[arXiv:1609.07088](https://arxiv.org/abs/1609.07088)，ICRA 2017）
- 数据侧与 [具身数据纵深](depth-embodied-data.md)（跨本体数据怎么聚合）串联；全身控制侧与 [BFM 纵深](depth-bfm.md)（一个 checkpoint 控全身）串联
- 若目标是"读一条示范就换任务"，那是 [ICL 纵深](depth-icl.md)；本路线关心的是"换身体"

## 谱系速览（一手来源）

下表按时间列出本方向的关键节点，数字均摘自论文摘要；工作名带链接的是站内已有详情页，其余直接读原文。

| 年份 | 工作 | 一手来源 | 对本路线的意义 |
|------|------|----------|----------------|
| 2017 | 模块化多机器人策略（Devin 等） | [arXiv:1609.07088](https://arxiv.org/abs/1609.07088)（ICRA 2017） | 策略拆成任务模块 × 机器人模块，可组合出未训练过的机器人–任务对；**本路线起点** |
| 2018 | NerveNet | [项目页](https://www.cs.toronto.edu/~tingwuwang/nervenet.html)（ICLR 2018） | 把智能体身体建成图、用 GNN 作策略；提出尺寸迁移与"残肢"迁移基准 |
| 2019 | [RoboNet](../wiki/entities/paper-rcl-1910-11215-robonet-large-scale-multi-robot-learning.md) | [arXiv:1910.11215](https://arxiv.org/abs/1910.11215)（CoRL 2019） | 7 个平台、1500 万帧；在留出的 Franka / Kuka 上微调，超过使用 4–20 倍数据的单机方法 |
| 2020 | Shared Modular Policies | [arXiv:2007.04976](https://arxiv.org/abs/2007.04976)（ICML 2020） | 每个执行器一个相同模块、只看局部传感并互传消息；一个策略控制不同骨架，泛化到未见形态 |
| 2022 | MetaMorph | [arXiv:2203.11931](https://arxiv.org/abs/2203.11931)（ICLR 2022） | Transformer 把身体结构当作额外输入，多形态预训练后零样本控制未见机器人 |
| 2023 | [Open X-Embodiment / RT-X](../wiki/entities/paper-open-x-embodiment.md) | [arXiv:2310.08864](https://arxiv.org/abs/2310.08864) | 21 家机构、22 种机器人、527 项技能；RT-X 借其他平台经验提升多台机器人能力（正迁移） |
| 2023 | Polybot | [arXiv:2307.03719](https://arxiv.org/abs/2307.03719) | 腕部相机 + 统一模块化代码对齐观测/动作，再用对比学习对齐表征；3 台机械臂约 60 h |
| 2024 | Mirage | [arXiv:2402.19249](https://arxiv.org/abs/2402.19249)（RSS 2024） | 部署时遮掉目标机器人、补画源机器人（cross-painting），零样本跨臂/跨夹爪 |
| 2024 | [Octo](../wiki/entities/paper-octo.md) | [arXiv:2405.12213](https://arxiv.org/abs/2405.12213) | 可插拔 token 输入与读出头，少样本适配新观测/新动作空间 |
| 2024 | CrossFormer | [arXiv:2408.11812](https://arxiv.org/abs/2408.11812) | 约 90 万条轨迹、20 种本体（单/双臂、轮式、四足、四旋翼）；无需手工对齐观测/动作空间，单权重持平各本体专才 |
| 2024 | HPT | [arXiv:2409.20537](https://arxiv.org/abs/2409.20537)（NeurIPS 2024） | 各本体本体感知与视觉先对齐成短 token 序列，再进共享 trunk；52 个数据集，未见任务微调提升 20% 以上 |
| 2024 | URMA | [arXiv:2409.06366](https://arxiv.org/abs/2409.06366) | 形态无关编码器/解码器，一个腿足运动策略跨本体，迁到未见平台（仿真与真机） |
| 2024 | RoVi-Aug | [arXiv:2409.03403](https://arxiv.org/abs/2409.03403)（CoRL 2024） | 生成模型合成不同机器人与视角的训练图像，零样本部署到未见机器人，成功率最高 +30% |
| 2024 | LAPA | [arXiv:2410.11758](https://arxiv.org/abs/2410.11758)（ICLR 2025） | VQ-VAE 学帧间潜在动作，在无动作标签视频上预训练 VLA；仅用人类视频也出现正迁移 |
| 2025 | Embodiment Scaling Laws | [arXiv:2505.05753](https://arxiv.org/abs/2505.05753)（CoRL 2025） | 程序生成约 1000 种本体；增加训练本体数比在固定本体上加数据泛化更广，零样本上 Go2 与 H1 |
---

## Stage 0 本体差异画像：先说清差在哪几层

**"跨本体"不是一个问题而是一族问题；先把源本体和目标本体的差异拆成五层，再决定哪些层交给接口、哪些交给训练、哪些只能交给目标机数据。**

### 前置知识
- 在一台机器人上完整训过并部署过一个策略
- 读得懂 URDF / MJCF：关节树、限位、质量惯量、执行器参数

### 核心问题
- **五层差异**：① 观测（相机位姿、视角、传感器种类）② 运动学（拓扑、DoF、连杆比例、闭链/髋轴结构）③ 动力学（质量分布、执行器力矩/带宽、PD 增益）④ 末端/工具（夹爪、灵巧手、工具头）⑤ 控制接口（动作语义、绝对/增量、控制频率）
- **迁移场景谱**（由近到远）：同形态跨配置（只换相机/夹爪）→ 换末端 → 近亲人形（关节近似一一对应）→ 异构人形 → 跨形态类别（人形↔四足↔臂）→ 人→机
- **"通用"的边界**：具身大模型的通用性是任务泛化，有效范围仍受动作空间、传感闭环与安全边界约束，不等于可以脱离本体定义
- 三条主路径的取舍：单本体重训 + 重定向（最稳最贵）、已有专家的高效后训练（如 Any2Any）、一开始就多本体联合训练（边际成本低、前期最贵）

### 推荐做什么
- 给自己的"源本体 → 目标本体"填一张五层差异表，每层标注：可由接口吸收 / 需训练吸收 / 必须目标机数据
- 用选型指南的决策树走一遍，写下自己落在哪条路径、为什么不是另外两条

### 推荐读什么
- [跨具身迁移（知识链汇总）](../wiki/overview/hub-cross-embodiment.md) — 本路线的知识枢纽：重定向、域随机与 Any2Any 类方法的选型与失败模式
- [跨具身策略迁移选型指南](../wiki/queries/cross-embodiment-transfer-strategy.md) — 单本体重训 / Any2Any / 多本体联合训练三路径的决策树与故障模式表
- [具身大模型与本体协同设计](../wiki/concepts/embodied-foundation-model-hardware-codesign.md) — "通用"是任务泛化而非脱离本体
- [Emergent Transfer](../wiki/entities/paper-emergent-transfer-cross-config.md) — 场景谱最近的一端：同形态跨配置，遗留示教在新机越过阈值 τ(T) 前无效
- [GEN-1 千手](../wiki/entities/generalist-gen1-thousand-hands.md) — 末端/工具轴：同一基座跨约 9000 末端变体（闭源自报）

### 学完输出什么
- 一张五层差异表 + 一句话定位："我的迁移落在场景谱第几档、走哪条路径"
- 能解释为什么"改完 URDF 直接上"会失败——重定向只解参考层，动力学差异仍要训练吸收

---

## Stage 1 统一接口：动作与状态空间怎么对齐

**多本体共用一个策略的前提是共用一套"语言"；接口抽象越高越好共享，越低越保真——这是本方向的第一个核心权衡。**

### 前置知识
- Stage 0 内容
- 正/逆运动学、末端位姿表示（SE(3)、相机系 vs 基座系）

### 核心问题
- **五类接口**：
  - **末端粗动作**：OXE / RT-X 把各数据集主要对齐到 7 维末端动作（位移 + 旋转 + 夹爪）——统一的是存储与粗接口，不是动力学
  - **规范槽位 + 掩码**：按身体语义把关节放进固定槽位，缺失关节用掩码屏蔽——XHugWBC 的 32 槽、GreenWAM 的 79 个解剖关节槽、Qwen-RobotManip 的 80 维 canonical state-action
  - **几何锚点**：用相机系可观测的腕/抓取几何作共享目标，再由翻译器出各本体命令（UCAG-P）；或把 TCP 坐标系投影进图像让 VLM 直接出增量（EASG）
  - **领域规范空间**：灵巧手用规范球面形变 + 级联 IK 映射到不同手（UHAS）；人手/灵巧手/夹爪用腕 SE(3) + 指关节槽（AdvDex）
  - **潜在动作**：从视频学与本体无关的 latent action，再解码到各本体（LAPA 用 VQ-VAE 学帧间离散潜动作；41 项 LAM 设计的实证见 Latent Actions Matter）
- **对齐还是不对齐**：Polybot 用腕部相机与统一代码库把观测/动作手工对齐，再用对比学习对齐内部表征；CrossFormer 则主张不做手工对齐，直接把 20 种本体的原生观测/动作交给 Transformer——两条路线的取舍是工程契约 vs 数据规模
- **观测层的外观差异**：机器人长得不一样本身就是分布偏移——Mirage 在部署时把目标机器人遮掉、补画成源机器人；RoVi-Aug 在训练时用生成模型合成不同机器人与视角
- **相机系 vs 基座系**：跨本体时基座系动作语义随安装位置变，相机系更可迁移但依赖外参
- **描述文件本身也是接口**：DoF 映射、链结构、碰撞近似若每次从 URDF 手工派生，接口会在工程层就不一致

### 推荐做什么
- 在两台手头本体上各实现一次"末端粗动作"与"规范槽位 + 掩码"两种接口，记录各自丢掉了什么信息（通常是手指、腰、冗余自由度）
- 写一份槽位定义表：每个槽的语义、单位、坐标系、缺失时的掩码规则——这张表就是后续所有混训的契约

### 推荐读什么
- [Open X-Embodiment](../wiki/entities/paper-open-x-embodiment.md) — 60+ 数据集、22 类本体的统一 schema；7 维末端粗动作的来源与局限
- [XHugWBC](../wiki/entities/paper-xhugwbc-cross-humanoid.md) — 32 槽关节语义空间 + 可控性掩码，异构人形的槽位接口范例
- [GreenWAM](../wiki/entities/paper-greenwam.md) — 20 类机器人映射到 79 个共享解剖关节槽，联合训练世界预测与策略
- [Qwen-RobotManip](../wiki/entities/qwen-robot-manip.md) — 80 维 canonical state-action + 相机系 EEF delta
- [UCAG-P](../wiki/entities/paper-ucag-p.md) · [EASG](../wiki/entities/paper-easg-vlm-native-action-model.md) — 相机系几何锚点两种落法
- [UHAS](../wiki/methods/uhas-unified-hand-action-space.md) · [AdvDex](../wiki/entities/paper-advdex.md) — 灵巧手层的两种统一动作空间（RL 球面 vs VLA 关节槽）
- [Latent Actions Matter](../wiki/entities/paper-latent-actions-matter.md) · [LAC-WM](../wiki/entities/paper-lac-wm-cross-embodiment-latent-action-world-model.md) — 潜在动作接口的实证与跨本体世界模型用法
- [URDD](../wiki/entities/paper-urdd-universal-robot-description-directory.md) — 把 URDF 派生的 DoF 映射、链结构等整理成带版本的模块目录
- 一手原文：[Polybot（arXiv:2307.03719）](https://arxiv.org/abs/2307.03719) · [CrossFormer（arXiv:2408.11812）](https://arxiv.org/abs/2408.11812) · [LAPA（arXiv:2410.11758）](https://arxiv.org/abs/2410.11758) · [Mirage（arXiv:2402.19249）](https://arxiv.org/abs/2402.19249) · [RoVi-Aug（arXiv:2409.03403）](https://arxiv.org/abs/2409.03403)

### 学完输出什么
- 一份接口选型便签：接口类型 × 能覆盖的本体范围 × 丢失的信息 × 需要的外参/标定
- 一张自己的槽位契约表，能直接给数据管线和训练代码共用

---

## Stage 2 形态感知架构：共享什么、本体专属什么

**架构的核心决定是"共享层多厚、本体层多厚"：共享层越厚迁移越划算，本体层越厚越退化回一机一策略。**

### 前置知识
- Stage 1 内容
- Transformer / GNN 的使用级直觉；会改策略网络的输入输出头

### 核心问题
- **共享骨干 + 本体专属头**：Octo 用可插拔读出头吃新动作空间；HPT 先把各本体的本体感知与视觉对齐成短 token 序列再进共享 trunk；GR00T N1 给每个本体配独立的状态/动作编码解码 MLP；X-WBC 把人类运动语义放进共享骨干、形态差异留给轻量本体模块——这一支的起点就是 Devin 2017 的"任务模块 × 机器人模块"
- **形态作为输入**：把关节树本身编码进网络。谱系是 NerveNet（GNN，ICLR 2018）→ Shared Modular Policies（每个执行器一个模块 + 消息传递，ICML 2020）→ MetaMorph（Transformer 把身体结构当输入，ICLR 2022）→ URMA（形态无关编码/解码）/ GET-Zero（本体图连通性作注意力偏置）；站内近作有 XHugWBC 的本体图网络、RecMorph 的运动学树 DFS 序 + 双向空间 RNN、Hodge 消息传递把四足建成胞腔复形——形态变了网络结构不用改
- **共享行为、各自执行**：CrossBFM 只共享高层行为潜空间，每台机器人保留自己的低层跟踪器——把跨本体复用放在行为接口上，而不是复制关节命令
- **历史即辨识**：LocoFormer 用 Transformer-XL 跨 episode 记忆在未见形态/电机故障下做部署期适应；UP-OSI（RSS 2017）用在线辨识器估参数再调策略——本体差异被当成"要从历史里推断的隐变量"
- **反过来用**：同一个多本体策略还能当设计工具（Shape Your Body 先训 URMA 多具身策略再沿价值梯度改机体参数；Transformer Transformer 统一共设计与跨具身控制）

### 推荐做什么
- 在同一批多本体数据上对比两种结构：共享骨干 + 每本体 MLP 头 vs 图/序列编码的形态感知网络，记录新增一个本体时各要改多少代码、训多少步
- 做一次"本体层厚度"消融：逐步把层从共享移到本体专属，画出平均性能与新本体适配成本的曲线

### 推荐读什么
- [Octo](../wiki/entities/paper-octo.md) — 读出头是迁移主接口：新相机、新动作空间不必重训整网
- [GR00T N1](../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — per-embodiment 状态/动作 MLP 投影到共享 DiT 隐空间
- [X-WBC](../wiki/entities/paper-x-wbc.md) — "运动语义可共享、物理执行不可共享"的切分；9 仿真人形联合训练、4 台真机同一接口
- [XHugWBC](../wiki/entities/paper-xhugwbc-cross-humanoid.md) · [RecMorph](../wiki/entities/paper-recmorph.md) · [Hodge 形态先验](../wiki/entities/paper-hodge-morphology-priors-quadruped-degradation.md) — 形态作为输入的三种编码
- [CrossBFM](../wiki/entities/paper-crossbfm-shared-latent-behavior.md) — 共享潜在行为空间 + 各本体专用跟踪器
- [LocoFormer](../wiki/entities/paper-locoformer.md) · [UP-OSI](../wiki/entities/paper-up-osi-universal-policy-online-sysid.md) — 历史上下文 / 在线辨识吸收本体差异
- [Shape Your Body](../wiki/entities/paper-shape-your-body-value-gradient-design.md) · [Transformer Transformer](../wiki/entities/paper-transformer-transformer.md) — 多具身策略反哺机体设计
- 一手原文（形态感知谱系）：[NerveNet](https://www.cs.toronto.edu/~tingwuwang/nervenet.html) · [Shared Modular Policies（arXiv:2007.04976）](https://arxiv.org/abs/2007.04976) · [MetaMorph（arXiv:2203.11931）](https://arxiv.org/abs/2203.11931) · [URMA（arXiv:2409.06366）](https://arxiv.org/abs/2409.06366) · [GET-Zero（arXiv:2407.15002）](https://arxiv.org/abs/2407.15002)
- 一手原文（共享骨干）：[HPT（arXiv:2409.20537）](https://arxiv.org/abs/2409.20537) — 52 个数据集上的 trunk 缩放研究

### 学完输出什么
- 一张架构选型表：共享层 / 本体层各放什么，新增本体的代码改动量与训练成本
- 能说清自己的系统"共享的是什么"——语义、行为、还是关节命令

---

## Stage 3 多本体混训：什么时候正迁移

**把多台机器人的数据塞进同一份训练并不自动带来跨本体能力；正迁移有条件、有阈值，还会偏科。**

### 前置知识
- Stage 2 内容
- 会写多数据集混合采样器，读得懂训练曲线里的分数据集指标

### 核心问题
- **谱系**：RoboNet（CoRL 2019）把 7 个平台、1500 万帧视觉经验汇成共享数据集，在留出的 Franka / Kuka 上微调即超过使用 4–20 倍数据的单机方法，但"相关子集"可以胜过更大的混合池 → OXE / RT-X（2023）在 22 种机器人上检验跨本体混训何时正迁移
- **本体数也是缩放轴**：Embodiment Scaling Laws（CoRL 2025）程序生成约 1000 种本体，报告增加训练本体数比在固定本体上加数据泛化更广，最佳策略零样本上 Go2 与 H1——腿足运动场景下的结论，操作侧仍待验证
- **阈值现象**：Emergent Transfer 报告新配置必须先靠自身数据越过任务依赖阈值 τ(T)，跨代共训才会涌现大幅收益——过早硬混遗留数据常零收益
- **扩形态分布**：XHugWBC 用同步调整几何、质量、惯量的物理一致随机形态；LocoFormer 用程序生成机器人——训练时见过的形态分布决定了零样本能走多远
- **通才 ↔ 专才循环**：EAGLE-WBC 用 generalist → specialist 微调 → DAgger 回蒸迭代，让单一策略跨 H1/G1/T1 等人形
- **偏科与配比**：数据量大的本体会主导梯度；按本体难度重加权、按本体分别报指标，而不是只看平均
- **无本体预训练**：Xiaomi-Robotics-1 先在 >100k h UMI 无本体数据上预训练，再用 ~10k h 跨本体数据后训练——把"本体"推迟到后训练阶段

### 推荐做什么
- 做一次最小混训实验：A 本体单训 / A+B 混训 / A+B 按难度重加权三档，按本体分别报成功率，确认是正迁移、零迁移还是负迁移
- 给目标本体画一条 standalone 数据量–成功率曲线，找到加入异本体数据开始有收益的拐点

### 推荐读什么
- [RoboNet](../wiki/entities/paper-rcl-1910-11215-robonet-large-scale-multi-robot-learning.md) — 多机器人共享经验的早期数据集；相关子集可胜过更大混合池
- [Open X-Embodiment 概念页](../wiki/concepts/open-x-embodiment.md) — 跨本体数据规范化与通用策略预训练
- [Emergent Transfer](../wiki/entities/paper-emergent-transfer-cross-config.md) — τ(T) 阈值与 phase-aware 采集规则
- [EAGLE-WBC](../wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md) — generalist→specialist→DAgger；仿真 5 机、真机 4 机
- [Xiaomi-Robotics-1](../wiki/entities/xiaomi-robotics-1.md) — 无本体 UMI 预训练 + 跨本体后训练
- [具身缩放律](../wiki/concepts/embodied-scaling-laws.md) — 数据规模、参数量与跨本体泛化的幂律讨论
- [具身数据纵深](depth-embodied-data.md) — 跨本体数据怎么聚合、清洗、配比
- 一手原文：[RoboNet（arXiv:1910.11215）](https://arxiv.org/abs/1910.11215) · [Open X-Embodiment（arXiv:2310.08864）](https://arxiv.org/abs/2310.08864) · [Towards Embodiment Scaling Laws in Robot Locomotion（arXiv:2505.05753）](https://arxiv.org/abs/2505.05753) · [Pushing the Limits of Cross-Embodiment Learning（arXiv:2402.19432）](https://arxiv.org/abs/2402.19432)（臂、四旋翼、四足、移动底座共训一个目标条件策略）

### 学完输出什么
- 一张分本体的混训消融表（单训 / 混训 / 重加权），能直接回答"混进来的数据到底帮没帮忙"
- 一份采集建议：目标本体先采多少条，再开始混异本体数据

---

## Stage 4 人→机迁移：最大跨度的跨本体

**人是数量最多、最便宜、也最"异构"的本体；这条线的全部难点在于把人的动作变成机器人能执行的东西，而不是把机器人假装成人。**

### 前置知识
- Stage 3 内容
- [动作重定向纵深](depth-motion-retargeting.md) Stage 0–2 水平（运动学 vs 动力学可行性）

### 核心问题
- **路线一：显式重定向**：人体动作 → 目标骨架参考 → 跟踪策略；X-Morph 把人体运动经 G1 再跨形态重定向 + 物理校正 + 跟踪蒸馏，给 Go2 / 六足 / B2-Z1 提供行为先验
- **路线二：重定向物体而非人手**：手机采集物体位姿与接触区间，对每个 URDF 做仿真可执行性判定——42 个 capture × 8 机器人中约 31% 通过、52% 失败，说明"人能做的机器人不一定能做"
- **路线三：本体无关表征**：EgoJEPA 从第一视角人类视频学与本体无关的潜在动作，不依赖手工重定向；LaST-HD 用动作条件世界模型把非配对人手与机器人轨迹对齐到共享潜空间
- **配对数据的强弱**：强 pair data（场景、逐帧时序、可 replay 动作严格对齐）才能把人的动力学拉进机器人域；弱 pair 只对齐任务语义
- **评测缺口**：人类视频 → 指定机器人本体视频的生成质量需要专门的诊断基准（H2R-Bench）
- **产业宣称**：OM-1 宣称仅用人类可穿戴示范、无遥操作/机上数据驱动跨工业臂与人形的策略——闭源自报，需独立验证

### 推荐做什么
- 选同一个任务，分别走"显式重定向"和"本体无关潜动作"两条线各跑一次，把失败按 **运动学不可达 / 动力学不可行 / 语义丢失** 三类计数
- 对自己的人类数据做一次可执行性筛查：多少比例在目标本体上物理可行——这个比例决定人类数据的真实"折算率"

### 推荐读什么
- [X-Morph](../wiki/entities/paper-xmorph.md) — 人体→非人形腿式：重定向 + 物理校正 + 特权跟踪蒸馏
- [重定向物体而非人手](../wiki/entities/paper-human-demos-object-retargeting.md) — 对每个 URDF 做仿真可执行性判定；约 31% 通过
- [EgoJEPA](../wiki/entities/paper-egojepa-egocentric-latent-action.md) — 第一视角人类视频 → 本体无关潜在动作
- [LaST-HD](../wiki/entities/paper-last-hd-latent-physical-reasoning.md) — 非配对人手与机器人轨迹对齐到共享潜式物理推理空间
- [强 pair data](../wiki/concepts/strong-pair-data.md) — 人–机配对数据的强弱分级
- [H2R-Bench](../wiki/entities/paper-h2r-bench.md) — 人类视频→指定机器人本体视频的跨本体诊断基准
- [Human2Dex](../wiki/entities/paper-human2dex-wrist-centric-dexterous.md) — 同一批人手示范分别重定向到两只不同灵巧手
- [OM-1（Reward AI）](../wiki/entities/reward-ai-om1.md) — 仅人类示范直驱跨本体的产业样本（闭源自报）
- 一手原文：[LAPA（arXiv:2410.11758）](https://arxiv.org/abs/2410.11758) — 仅用人类操作视频预训练潜动作 VLA 也出现正迁移

### 学完输出什么
- 一条"人类数据 → 目标本体可执行轨迹"的最短链路，并标出哪一环最脆
- 一份人类数据折算率记录：采集量 × 可执行比例 × 下游增益

---

## Stage 5 新本体快速适配与验收

**迁移的最后一公里是"新机器人到货那天要多少数据、多少算力、多大风险"；数字要和源本体退化、负迁移、安全护栏放在一起报。**

### 前置知识
- Stage 4 内容
- LoRA / 参数高效微调；真机部署的安全层（见 [Sim2Real 纵深](depth-sim2real.md)）

### 核心问题
- **冻结 + 低秩补丁**：Any2Any 把差距拆成无梯度运动学对齐 + 动力学敏感层 LoRA，约 1% 全量训练成本迁到新人形；近亲骨架上 SONIC-Transfer 冻得更死（闭式 codec + 约 0.25% 参数）；TRACE 把 SONIC 从 G1 迁到 H2
- **按情境生成适配参数**：MorphAct 为共享 VLA 生成低秩参数更新，而不是每个场景单独微调
- **少样本端侧适配**：Gemini Robotics On-Device 博客称数小时、<200 例量级适配新双臂本体——自报数字
- **上下文适应**：不改权重、靠部署历史适应新本体（LocoFormer）；与 [ICL 纵深](depth-icl.md) 的机制选型同构
- **验收三件事**：① 留出本体（训练从未见过）上的成绩 ② 源本体是否退化（catastrophic forgetting）③ 新机首日安全护栏——力矩/速度限幅先于任何策略上机
- **跨篇数字不可横比**：各工作的本体集合、参考动作语料、指标定义都不同，选型按场景谱与预算，不按榜单

### 推荐做什么
- 给一台新本体做一次完整适配记账：目标机数据量、GPU 小时、源本体指标变化、留出任务成功率、首次上机的安全措施
- 写一份自己的跨本体评测协议：哪些本体参与训练、哪些留出、每个本体分别报什么指标、负迁移怎么判定

### 推荐读什么
- [Any2Any](../wiki/entities/paper-any2any-cross-embodiment-wbt.md) — 运动学对齐 + 动力学 LoRA，约 1% 成本迁新人形
- [SONIC-Transfer](../wiki/entities/paper-sonic-transfer.md) · [TRACE](../wiki/entities/paper-trace-humanoid-transfer.md) — 近亲骨架冻结迁移与 G1→H2 两阶段课程
- [MorphAct](../wiki/entities/paper-morphact.md) — 按情境生成低秩参数更新的共享 VLA
- [Gemini Robotics](../wiki/entities/gemini-robotics.md) — On-Device 快速跨本体适配（闭源自报）
- [Whole-Body Tracking Pipeline](../wiki/concepts/whole-body-tracking-pipeline.md) — 跨具身迁移在 WBT 全流水线中的位置
- [具身模型测评纵深](depth-embodied-eval.md) — 留出集划分与 sim↔real 校准

### 学完输出什么
- 一份新本体适配记账单，能直接回答"再来一台新机器人要花多少"
- 一份跨本体评测协议文档，含留出本体、源本体退化与负迁移判据

---

## 快速入口汇总

| 阶段 | 核心问题 | 知识页入口 |
|------|---------|-----------|
| Stage 0 | 本体差异分几层、迁移落在哪档 | [跨具身迁移知识链](../wiki/overview/hub-cross-embodiment.md) · [选型指南](../wiki/queries/cross-embodiment-transfer-strategy.md) |
| Stage 1 | 动作/状态接口怎么对齐 | [Open X-Embodiment](../wiki/entities/paper-open-x-embodiment.md) · [XHugWBC](../wiki/entities/paper-xhugwbc-cross-humanoid.md) |
| Stage 2 | 共享什么、本体专属什么 | [X-WBC](../wiki/entities/paper-x-wbc.md) · [Octo](../wiki/entities/paper-octo.md) |
| Stage 3 | 多本体混训何时正迁移 | [Emergent Transfer](../wiki/entities/paper-emergent-transfer-cross-config.md) · [EAGLE-WBC](../wiki/entities/paper-notebook-embodiment-aware-generalist-specialist-distillat.md) |
| Stage 4 | 人→机怎么迁 | [X-Morph](../wiki/entities/paper-xmorph.md) · [EgoJEPA](../wiki/entities/paper-egojepa-egocentric-latent-action.md) |
| Stage 5 | 新本体适配与验收 | [Any2Any](../wiki/entities/paper-any2any-cross-embodiment-wbt.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制 → 物理智能全栈成长路线](motion-control.md)
- 相关纵深路线（按主题邻近，完整目录见 [路线总览](motion-control.md#depth-optional-index)）：
  - [动作重定向](depth-motion-retargeting.md) — 参考层的跨骨架映射，是本路线 Stage 4 的上游
  - [BFM](depth-bfm.md) — 一个 checkpoint 控全身；跨本体是 BFM 的下一道题
  - [Sim2Real](depth-sim2real.md) — 同一本体的仿真↔真机，常与跨本体串联
  - [具身数据](depth-embodied-data.md) — 跨本体数据的聚合与配比
  - [VLA](depth-vla.md) — 操作侧跨本体预训练的主要载体
  - [ICL](depth-icl.md) — 换任务 vs 换身体；部署期适应机制同构

## 参考来源

- [跨具身迁移知识链汇总](../wiki/overview/hub-cross-embodiment.md)
- [跨具身策略迁移选型指南](../wiki/queries/cross-embodiment-transfer-strategy.md)
- [Learning Modular Neural Network Policies for Multi-Task and Multi-Robot Transfer（arXiv:1609.07088）](https://arxiv.org/abs/1609.07088) — Devin、Gupta、Darrell、Abbeel、Levine；任务模块 × 机器人模块，本路线起点
- [NerveNet: Learning Structured Policy with Graph Neural Networks（ICLR 2018，项目页）](https://www.cs.toronto.edu/~tingwuwang/nervenet.html) — Wang、Liao、Ba、Fidler；图结构策略与结构迁移基准
- [RoboNet: Large-Scale Multi-Robot Learning（arXiv:1910.11215）](https://arxiv.org/abs/1910.11215) — 多机器人共享数据集
- [One Policy to Control Them All: Shared Modular Policies for Agent-Agnostic Control（arXiv:2007.04976）](https://arxiv.org/abs/2007.04976) — Huang、Mordatch、Pathak；ICML 2020
- [MetaMorph: Learning Universal Controllers with Transformers（arXiv:2203.11931）](https://arxiv.org/abs/2203.11931) — Gupta、Fan、Ganguli、Fei-Fei；ICLR 2022
- [Open X-Embodiment: Robotic Learning Datasets and RT-X Models（arXiv:2310.08864）](https://arxiv.org/abs/2310.08864) — 跨本体数据聚合与 RT-X 正迁移检验
- [Polybot: Training One Policy Across Robots While Embracing Variability（arXiv:2307.03719）](https://arxiv.org/abs/2307.03719) — Yang、Sadigh、Finn
- [Mirage: Cross-Embodiment Zero-Shot Policy Transfer with Cross-Painting（arXiv:2402.19249）](https://arxiv.org/abs/2402.19249) — RSS 2024
- [Pushing the Limits of Cross-Embodiment Learning for Manipulation and Navigation（arXiv:2402.19432）](https://arxiv.org/abs/2402.19432)
- [GET-Zero: Graph Embodiment Transformer for Zero-shot Embodiment Generalization（arXiv:2407.15002）](https://arxiv.org/abs/2407.15002)
- [Scaling Cross-Embodied Learning: One Policy for Manipulation, Navigation, Locomotion and Aviation（arXiv:2408.11812）](https://arxiv.org/abs/2408.11812) — CrossFormer
- [RoVi-Aug: Robot and Viewpoint Augmentation for Cross-Embodiment Robot Learning（arXiv:2409.03403）](https://arxiv.org/abs/2409.03403) — CoRL 2024
- [One Policy to Run Them All: an End-to-end Learning Approach to Multi-Embodiment Locomotion（arXiv:2409.06366）](https://arxiv.org/abs/2409.06366) — URMA
- [Scaling Proprioceptive-Visual Learning with Heterogeneous Pre-trained Transformers（arXiv:2409.20537）](https://arxiv.org/abs/2409.20537) — HPT；NeurIPS 2024
- [Latent Action Pretraining from Videos（arXiv:2410.11758）](https://arxiv.org/abs/2410.11758) — LAPA；ICLR 2025
- [Towards Embodiment Scaling Laws in Robot Locomotion（arXiv:2505.05753）](https://arxiv.org/abs/2505.05753) — CoRL 2025
