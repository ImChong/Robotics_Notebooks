# 路线（纵深）：如果目标是模型架构（骨干族谱 → 感知编码 → 动作头 → 多模态基座 → 规模与部署）

**摘要**：面向"想知道一条机器人策略该用什么网络、多宽多深、动作怎么出"的纵深路线，从"由控制回路反推函数族"的选型判据出发，沿 MLP / CNN / RNN / Transformer / SSM 骨干族谱 → 感知编码器与表征接口 → 动作头（单步回归 / 动作块 / 生成式）→ 多模态基座（VLA / WAM / 统一 token / 双系统）→ 规模、延迟与部署，按 Stage 0–5 串通核心页；本路线是 [运动控制主路线](motion-control.md) 的一条分支，与 [具身数据纵深](depth-embodied-data.md) 构成"喂什么 / 用什么装"的姊妹路线。

## 路线一览

```mermaid
flowchart LR
  S0["<b>Stage 0</b><br/>选型判据<br/><em>回路反推函数族 · 分层频率</em>"]
  S1["<b>Stage 1</b><br/>骨干族谱<br/><em>MLP · CNN · RNN · Transformer · SSM</em>"]
  S2["<b>Stage 2</b><br/>感知编码<br/><em>视觉骨干 · 表征接口 · 多模态融合</em>"]
  S3["<b>Stage 3</b><br/>动作头<br/><em>单步回归 → chunk → 生成式</em>"]
  S4["<b>Stage 4</b><br/>多模态基座<br/><em>VLA / WAM · 统一 token · 双系统</em>"]
  S5["<b>Stage 5</b><br/>规模与部署<br/><em>缩放 · 延迟预算 · 蒸馏量化</em>"]

  S0 --> S1 --> S2 --> S3 --> S4 --> S5

  classDef stage fill:#142a3a,stroke:#e67e22,stroke-width:2px,color:#fff
  class S0,S1,S2,S3,S4,S5 stage
```

## 这条路径怎么用

- 目标读者是已经会训一个策略、但每次读论文 Method 的"网络结构"段落都只能照抄的人——本路线把散落在概念页、方法页与对比页里的架构知识串成一条可选型的链
- 这条路线回答的是 **"用什么函数族装这个映射"**，不回答"喂什么数据"（那是 [具身数据纵深](depth-embodied-data.md)）、"用什么目标训"（那是 [RL 运动控制](depth-rl-locomotion.md) 与 [模仿学习](depth-imitation-learning.md) 纵深）、"怎么证明它更好"（那是 [具身测评纵深](depth-embodied-eval.md)）
- **最容易犯的错是把架构当主要瓶颈**：人形真机最强的低层策略至今常是 2–3 层、256–512 宽的 MLP，瓶颈在观测、延迟、执行器与迁移；Stage 0 先把这条判断钉死，后面几个 Stage 才不会变成"追新骨干"
- 架构选择几乎总是被 **控制频率与时延预算** 反向约束：同一条栈里，慢语义层可以是几十亿参数的 VLM，快跟踪层必须能在毫秒级跑完
- 每个阶段都有前置知识、核心问题、推荐做什么、推荐读什么、学完输出什么

**和主路线的关系：**
- 本路线横切主路线 L5（RL 与模仿学习）到 L7（出口层）：L5 之前先用小 MLP 打通闭环，读完 Stage 0–1 足够；进入操作、语言条件与基础模型时再走 Stage 2–4
- 起点取 **反向传播（Rumelhart et al., Nature 1986）**——多层网络可训练之后，"选哪种函数族"才第一次成为工程问题
- 与 [VLA](depth-vla.md)、[BFM](depth-bfm.md)、[WAM](depth-wam.md) 三条纵深的关系是"横切 vs 纵深"：那三条讲某一类模型解决什么问题，本路线讲这些模型内部的结构件怎么挑、怎么拼、怎么塞进时延预算

---

## Stage 0 选型判据：从控制回路反推函数族

**先写清"这个网络挂在哪一层、多久被调用一次、输入在部署时是否可得"，再谈骨干；顺序反过来就会得到一个跑不进控制环的漂亮模型。**

### 前置知识
- Python + PyTorch 能跑通一次训练与推理
- 知道一条 RL / IL 策略的闭环长什么样（观测 → 网络 → 动作 → 环境）
- 对 PD 控制、关节目标与力矩指令的区别有直觉

### 核心问题
- 这个网络属于 **大脑（做什么）/ 小脑（怎么做）/ 脊髓（保命）** 三层里的哪一层？不同层的时延预算差两到三个数量级
- 输入里哪些是 **部署可得**、哪些是 **仅训练可得的特权信息**？后者决定了要不要做 teacher–student 双网结构
- 输出是单步关节目标、一段动作块，还是 token 序列？动作表示一变，骨干与解码器全要跟着换
- 为什么"模型越大越强"在人形低层控制上长期不成立——瓶颈更常在观测、奖励、执行器与 sim2real

### 推荐做什么
- 给自己手上的任务画一张 **分层频率表**：每层的调用周期、输入来源、允许的最坏推理时延、失效时的降级动作
- 拿 3 篇本方向论文，只读 Method 的结构段落，把层数 / 宽度 / 激活 / 历史长度 / 动作 horizon 抄成一张对照表

### 推荐读什么
- [人形与腿式策略的网络架构](../wiki/concepts/humanoid-policy-network-architecture.md) — 本路线的总纲：从浅层 MLP、AMP 判别器、MoE 到 Transformer/Diffusion chunk 与 VLA/WAM 的代际对照表
- [人形运控策略的观测输入](../wiki/concepts/humanoid-policy-observation-inputs.md) — 五类输入按"部署是否可得"分类，决定架构的输入端设计
- [具身三层控制架构](../wiki/concepts/embodied-three-layer-control-architecture.md) — 大脑 / 小脑 / 脊髓的时延预算与故障降级
- [控制频率与推理频率解耦](../wiki/concepts/control-inference-frequency-decoupling.md) — 低频策略与高频执行环之间的三类接口
- [神经反馈控制器](../wiki/concepts/neural-feedback-controller.md) — 把 policy 读成状态反馈律 $\pi(x)$，建立推理算力直觉
- [深度学习基础](../wiki/concepts/deep-learning-foundations.md) · [反向传播](../wiki/concepts/backpropagation.md) — 函数逼近与可微计算图底座

### 学完输出什么
- 一张自己场景的分层频率表（层 × 周期 × 输入 × 时延上限 × 降级）
- 一句话判断：这个任务真正的瓶颈是架构表达力，还是观测 / 奖励 / 迁移

---

## Stage 1 骨干族谱：MLP / CNN / RNN / Transformer / SSM

**四类骨干的差别可以压成三维——归纳偏置、长程依赖、推理复杂度；选型就是在这三维上对齐任务与时延预算。**

### 前置知识
- Stage 0 内容
- 会手写一个 MLP 策略并训到收敛

### 核心问题
- 归纳偏置换算力：CNN 的局部性与权值共享在小数据 / 边缘部署上仍占优，注意力弱偏置要靠数据量换回来
- 长程依赖的四种做法：RNN 递推、CNN 深堆叠、注意力 $O(1)$ 路径、选择性 SSM 的状态压缩；对应的推理复杂度分别是 $O(n)$、近线性、$O(n^2)$、近线性
- 历史观测该怎么进网络：帧堆叠、GRU/LSTM 隐状态，还是上下文窗口？这直接决定部署时的状态管理复杂度
- 为什么 benchmark 赢不等于机载赢——算子成熟度、显存带宽与量化友好度是真实约束

### 推荐做什么
- 用 [RNN vs CNN vs Transformer vs Mamba](../wiki/comparisons/rnn-cnn-transformer-mamba.md) 的四维表，给自己的两个场景（高频本体控制、带图像的操作）各选一个骨干并写明理由
- 在同一任务上把"帧堆叠 MLP"与"小 Transformer"各训一版，对比样本效率与单步推理耗时

### 推荐读什么
- [RNN vs CNN vs Transformer vs Mamba](../wiki/comparisons/rnn-cnn-transformer-mamba.md) — 四类骨干在长程 / 并行 / 复杂度 / 归纳偏置上的对照
- [Transformer](../wiki/concepts/transformer.md) · [多头注意力](../wiki/concepts/multi-head-attention.md) — 现代骨干的通用底座与核心交互算子
- [卷积神经网络](../wiki/concepts/convolutional-neural-network.md) · [通道与空间注意力](../wiki/methods/channel-spatial-attention.md) — 卷积路线与 Transformer 之前的注意力增强
- [状态空间模型（SSM）](../wiki/concepts/state-space-model-ssm.md) — 选择性 SSM 如何在近线性复杂度下拿到长程能力
- [U-Net](../wiki/methods/unet.md) — 编码器–解码器 + 跳跃连接，后来成为扩散策略的常用去噪骨干
- [深度学习优化器对比](../wiki/comparisons/deep-learning-optimizers.md) — 换骨干往往要连带换优化器与学习率策略

### 学完输出什么
- 一张"归纳偏置 × 长程 × 并行度 × 推理复杂度"四维选型表，能对新骨干直接填格
- 一组自己测出的延迟数字：同一输入维度下，MLP / 小 Transformer 在目标硬件上的单步耗时

---

## Stage 2 感知编码：视觉骨干与表征接口

**感知侧的架构决策有两个：用哪种骨干提特征，以及这些特征以什么形式交给策略——后者常常比前者更影响成败。**

### 前置知识
- Stage 1 内容
- 跑过一次图像分类 / 检测或分割训练，知道 backbone 与 head 的分工

### 核心问题
- CNN 还是 ViT：小数据与高吞吐机载部署 vs 大数据规模化与全局注意力，判据是数据量、分辨率吞吐与下游任务
- 表征接口的三条路：端到端联合训练、冻结预训练骨干、机器人专用预训练表征（R3M / VC-1 / DINOv2）——样本效率与域差距的三角取舍
- 策略真正需要的表征往往不可读：感知行走里 encoder 输出的是地形摘要向量，而不是可视化的高度图
- 多模态怎么并进来：视觉 + 语言的语义对齐、视觉 + 触觉的时空对齐，各自的融合位置（早融合 / 晚融合 / 交叉注意力）不同
- token 数是隐藏的时延旋钮：分辨率、patch 大小、视角数都会直接乘进注意力开销

### 推荐做什么
- 在同一操作任务上对比"冻结 DINOv2 特征 + 小策略头"与"端到端小 CNN"，记录样本效率与过拟合表现
- 统计自己视觉栈的 token 预算：视角数 × patch 数 × 历史帧数，算出注意力部分的理论开销

### 推荐读什么
- [视觉骨干](../wiki/concepts/vision-backbones.md) · [Vision Transformer](../wiki/concepts/vision-transformer.md) · [CNN vs ViT 骨干选型](../wiki/comparisons/cnn-vs-vit-backbones.md)
- [策略的视觉表征](../wiki/concepts/visual-representation-for-policy.md) — 端到端 / 冻结骨干 / 机器人专用表征三条路径的取舍
- [地形潜表征](../wiki/concepts/terrain-latent-representation.md) — 编码器输出不必可读，只需支撑正确落脚
- [生成式视觉预训练](../wiki/concepts/generative-vision-pretraining.md) · [视觉基础模型趋势](../wiki/concepts/visual-foundation-model-trends.md)
- [多模态基础](../wiki/concepts/multimodality-basics.md) · [视觉–语言特征融合](../wiki/concepts/vision-language-feature-fusion.md) · [视触融合](../wiki/concepts/visuo-tactile-fusion.md)
- [具身感知六种空间表征](../wiki/concepts/embodied-perception-six-spatial-representations.md) — 2D / 深度 / 点云 / 占据 / 语义 / 隐式各自回答什么问题
- [视觉骨干知识链枢纽](../wiki/overview/hub-vision-backbone.md) — 骨干 → 检测分割头 → 策略输入的完整链路

### 学完输出什么
- 一份感知侧选型记录：骨干、是否冻结、特征维度、token 预算与实测吞吐
- 能说清自己策略"看到的"到底是什么表征，以及它在域变化下最先坏在哪

---

## Stage 3 动作头：从单步回归到动作块与生成式策略

**输出端的结构比输入端更决定行为形态：单步高斯回归会把多模态演示平均成中间值，动作块与生成式头就是为解决这件事而来。**

### 前置知识
- Stage 2 内容
- 训过一次行为克隆，见过"演示里有两种走法、策略学成中间那条"的现象

### 核心问题
- 为什么单步确定性 / 单峰高斯头在人类演示上会塌掉，扩散与流匹配头如何表达多模态动作分布
- 动作块（action chunking）解决的两件事：长时序误差累积，以及慢推理与快控制之间的时域错配
- 滚动执行的两种语义不可混读：预测一段只执行前缀再重规划（receding horizon），与重叠动作序列的时间集成（ACT 的 temporal ensemble）
- 训练期架构 ≠ 部署期架构：teacher–student 与非对称 actor–critic 让特权信息只出现在训练侧
- 多技能怎么装：multi-expert / MoE 门控是可读的技能分解基线，与"一个大骨干 + 条件输入"是两条路

### 推荐做什么
- 在一个有多模态演示的任务上，把动作头从单步回归换成扩散头（或流匹配头），对比成功率与动作平滑度
- 扫描 chunk 长度与执行前缀比例，画出"成功率 / 时延 / 抖动"三条曲线，定出自己场景的工作点

### 推荐读什么
- [行为克隆](../wiki/methods/behavior-cloning.md) · [Transformer 行为克隆](../wiki/methods/bc-with-transformer.md) — 从 MLP 头到序列骨干的第一跳
- [Action Chunking](../wiki/methods/action-chunking.md) — 一次输出多步的机制拆解：延迟观测条件化与隐式集成
- [滚动执行](../wiki/concepts/receding-horizon-policy-execution.md) — receding horizon 与 temporal ensemble 的区别
- [扩散模型](../wiki/concepts/diffusion-model.md) · [Diffusion Policy](../wiki/methods/diffusion-policy.md) — 多步去噪生成动作序列的表达力与代价
- [特权训练](../wiki/concepts/privileged-training.md) · [Teacher–Student 与 DAgger 训练](../wiki/methods/teacher-student-dagger-training.md) — 双网结构与蒸馏
- [AMP 奖励](../wiki/methods/amp-reward.md) — 判别器也是一个要设计的网络：输入取状态转移，宽深与稳定性强相关
- [人形与腿式策略的网络架构](../wiki/concepts/humanoid-policy-network-architecture.md) 的 multi-expert / MoE 小节

### 学完输出什么
- 一份动作头决策记录：动作表示、chunk 长度、执行前缀、推理步数与实测控制带宽
- 能解释自己策略的抖动来自哪一层：动作头分布、chunk 拼接，还是下游跟踪器

---

## Stage 4 多模态基座：VLA / WAM 的结构件与双系统

**到这一层，架构问题变成"怎么把一个预训练 VLM 接上机器人"：token 化方案、动作头形式、以及慢–快两套模型如何分工。**

### 前置知识
- Stage 3 内容
- 了解一个 VLM 的输入输出形态（图像 patch token + 文本 token → 自回归解码）

### 核心问题
- 统一 token 化：视觉 patch、语言词元与离散化动作编进同一嵌入空间，换来纯序列建模的简洁，代价是动作离散化精度与序列长度
- 动作头的两条路：离散 action token 自回归解码 vs 连续 flow / 扩散动作头；前者复用 VLM 解码栈，后者动作精度与平滑度更好
- 骨干强度 vs 头复杂度：强 VLM 底座配简单 MLP 动作头也能打到很强的控制性能，说明"头很花"未必是收益来源
- 读出头与跨本体适配：块状注意力 + 独立读出头让新相机、新动作空间可以少样本接入
- 双系统 / 慢–快分层：低频语义推理 + 高频跟踪执行，两套模型之间的接口是动作块、子目标还是残差
- WAM 把未来预测并进骨干：训练期多一路视频 / 潜空间监督，推理期是否还生成未来，是延迟的分水岭

### 推荐做什么
- 挑 3 个开源 VLA（例如 RT 系、Octo、π₀ 一族），画出各自的"输入 token 构成 → 骨干 → 动作头 → 输出频率"四段结构图
- 为自己的平台写一张接入判断表：能否容忍 5–10 Hz 的语义层，低层跟踪器由谁提供

### 推荐读什么
- [VLA](../wiki/methods/vla.md) · [Robotics Transformer（RT 系列）](../wiki/methods/robotics-transformer-rt-series.md) — 语言条件序列预测到 VLM 接入的基线谱系
- [统一多模态 Token](../wiki/methods/unified-multimodal-tokens.md) — 把视觉 / 语言 / 动作编进同一空间的架构趋势
- [Octo](../wiki/methods/octo-model.md) — 块状注意力 Transformer + 独立读出头，少样本适配新相机与动作空间
- [π₀](../wiki/methods/π0-policy.md) · [π₀.₇](../wiki/methods/pi07-policy.md) — 流匹配动作头与多模态提示对齐异质数据
- [StarVLA](../wiki/methods/star-vla.md) — 强 VLM 底座 + 简单 MLP 动作头的极简基准，用来校准"复杂头是否必要"
- [Foundation Policy](../wiki/concepts/foundation-policy.md) · [行为基础模型（BFM）](../wiki/concepts/behavior-foundation-model.md) — 操作向与全身控制向两类基座
- [World Action Models](../wiki/concepts/world-action-models.md) · [潜空间想象](../wiki/concepts/latent-imagination.md) — 未来预测进骨干后的结构变化
- [VLM / VLN / VLA / VLX / 世界模型分类学](../wiki/comparisons/vlm-vln-vla-vlx-world-model-taxonomy.md) · [具身大模型选型闭环枢纽](../wiki/overview/hub-embodied-foundation-model.md)

### 学完输出什么
- 三张主流 VLA 的结构拆解图，能指出各自的 token 预算与延迟来源
- 一份自己平台的分层方案：语义层模型规模与频率、低层跟踪器接口、失效时的降级路径

---

## Stage 5 规模、效率与部署：让结构活在时延预算里

**架构的最后一关不在论文表格里，而在机载芯片上：参数量、token 数与去噪步数最终都会折算成一条控制带宽。**

### 前置知识
- Stage 4 内容
- 部署过一次模型到机载设备（ONNX / TensorRT / MNN 任一路径）

### 核心问题
- 缩放该往哪加：参数、数据还是本体多样性？幂律在具身数据上的成立范围与已知反例
- 实时性 ↔ 泛化的可达边界：模型规模、多模态跨度与世界模型推演步长共同决定推理时延，进而决定控制带宽
- 压缩四件套：蒸馏、量化、算子替换与异步执行，各自牺牲什么（精度 / 平滑度 / 最坏时延）
- 部署栈选型：ONNX 作格式契约，ORT / TensorRT / MNN 按硬件与算子覆盖选，量化友好度反过来影响骨干选择
- 适配而非重训：LoRA 一类低秩适配如何用极小参数量改动力学敏感模块
- 架构不只在策略里：执行器网络把电机非线性塞进一个小网络，是"用网络补模型误差"的典型位置
- 上线之后：权重与固件分通道发布、签名验签与 A/B 回滚，把模型当成一个可回退的版本化产物

### 推荐做什么
- 给自己的栈做一次时延归因：感知编码 / 骨干前向 / 动作头去噪 / 通信各占多少毫秒，找出真正的大头
- 选一条压缩路径（蒸馏或量化）做一次 A/B：记录成功率、动作平滑度与最坏时延三项，而不是只看平均时延

### 推荐读什么
- [具身规模法则](../wiki/concepts/embodied-scaling-laws.md) — 数据 / 参数 / 泛化的幂律关系与边界
- [实时性与泛化取舍](../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) — 规模与推演步长如何压住控制带宽
- [ONNX Runtime vs MNN vs TensorRT](../wiki/comparisons/onnxruntime-vs-mnn-vs-tensorrt.md) — 机载推理栈选型
- [LoRA](../wiki/concepts/lora.md) — 低秩适配用于大模型的低成本模块级微调
- [执行器网络](../wiki/methods/actuator-network.md) — 网络在 sim2real 链路上的另一个落点
- [具身大模型与本体协同设计](../wiki/concepts/embodied-foundation-model-hardware-codesign.md) — 分层频率与本体定义权
- [边缘–云协同](../wiki/concepts/edge-cloud-robotics.md) · [模型版本管理与 OTA](../wiki/concepts/model-versioning-ota.md) — 上线与回退侧的工程约束

### 学完输出什么
- 一份时延归因报告与压缩前后的三项对比（成功率 / 平滑度 / 最坏时延）
- 一句话结论：在当前硬件上，这条栈的架构上限由什么决定——显存、带宽、去噪步数还是通信

---

## 快速入口汇总

| 阶段 | 核心问题 | 知识页入口 |
|------|---------|-----------|
| Stage 0 | 从控制回路反推函数族 | [人形与腿式策略的网络架构](../wiki/concepts/humanoid-policy-network-architecture.md) |
| Stage 1 | 骨干族谱与三维选型 | [RNN vs CNN vs Transformer vs Mamba](../wiki/comparisons/rnn-cnn-transformer-mamba.md) |
| Stage 2 | 感知编码与表征接口 | [策略的视觉表征](../wiki/concepts/visual-representation-for-policy.md) |
| Stage 3 | 动作头与动作块 | [Action Chunking](../wiki/methods/action-chunking.md) · [Diffusion Policy](../wiki/methods/diffusion-policy.md) |
| Stage 4 | 多模态基座结构件 | [统一多模态 Token](../wiki/methods/unified-multimodal-tokens.md) · [VLA](../wiki/methods/vla.md) |
| Stage 5 | 规模、时延与部署 | [实时性与泛化取舍](../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) |

## 和其他页面的关系

- 完整成长路线参考：[主路线：运动控制算法工程师成长路线](motion-control.md)
- 相关纵深路线（按主题邻近，完整目录见 [路线总览](motion-control.md#depth-optional-index)）：
  - [VLA](depth-vla.md) — 本路线 Stage 4 结构件在语义策略方向的完整展开
  - [WAM](depth-wam.md) — 把未来预测并进骨干后的 Cascaded / Joint 两条结构主线
  - [BFM](depth-bfm.md) — 全身控制向基座的骨干与技能空间设计
  - [模仿学习](depth-imitation-learning.md) — 动作头与数据形态的配套关系
  - [具身数据](depth-embodied-data.md) — 架构容量要由数据供给撑住，二者互为上限
  - [具身测评](depth-embodied-eval.md) — 架构改动要靠分层评测证伪，别只看单一成功率

## 参考来源

- [人形与腿式策略的网络架构](../wiki/concepts/humanoid-policy-network-architecture.md) — 代际演化表与"低层仍是小 MLP"的判断
- [RNN vs CNN vs Transformer vs Mamba](../wiki/comparisons/rnn-cnn-transformer-mamba.md) — 骨干三维对照
- [策略的视觉表征](../wiki/concepts/visual-representation-for-policy.md) 与 [视觉骨干知识链枢纽](../wiki/overview/hub-vision-backbone.md) — 感知侧接口
- [Action Chunking](../wiki/methods/action-chunking.md) 与 [滚动执行](../wiki/concepts/receding-horizon-policy-execution.md) — 输出端结构
- [统一多模态 Token](../wiki/methods/unified-multimodal-tokens.md) 与 [具身大模型选型闭环枢纽](../wiki/overview/hub-embodied-foundation-model.md) — 多模态基座
- [实时性与泛化取舍](../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) 与 [具身规模法则](../wiki/concepts/embodied-scaling-laws.md) — 规模与时延边界
- Rumelhart, Hinton & Williams, *Learning representations by back-propagating errors*, Nature 1986 — 本路线的起点里程碑（[归档](../sources/papers/rumelhart_backprop_learning_representations_nature_1986.md)）
