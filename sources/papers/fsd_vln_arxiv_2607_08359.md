# FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation（arXiv:2607.08359）

> 来源归档（ingest）

- **标题：** FSD-VLN: Fast-Slow Dual-System Modeling for Aerial Long-Horizon Vision-Language Navigation
- **类型：** paper / VLN / aerial-navigation / UAV / dual-system / DiT / low-latency
- **来源：** arXiv abs / PDF / HTML（v1，2026-07-09）
- **原始链接：**
  - <https://arxiv.org/abs/2607.08359>
  - PDF：<https://arxiv.org/pdf/2607.08359>
  - HTML：<https://arxiv.org/html/2607.08359v1>
- **作者：** Xueke Zhu\*、Qingyan Meng\*、Liutao Yu、Wei Zhang、Zhengyu Ma、Huihui Zhou、Yonghong Tian（\*同等贡献）
- **机构：** 鹏城实验室（Peng Cheng Laboratory）；中国科学院深圳先进技术研究院（SIAT）；北京大学（Peking University）
- **入库日期：** 2026-08-16
- **一句话说明：** 空中长程 VLN 的快慢双系统：慢路冻结 VLM 写语义缓冲，快路用 GR00T N1 系 DiT 异步生成飞行动作；仿真未见场景相对自复现 OpenFly 的 SR 约 2.7×，单步推理与任务总时长约减半；截至入库日确认未开源、无真机。

## 开源状态（核查 2026-08-16）

- **确认未开源 / 无项目页：** arXiv abs、HTML、PDF 均未列出 GitHub、Hugging Face、项目页或「code will be released」外链；公开 Web 检索未发现官方同名可运行仓。
- **复现边界：** 方法依赖 **GR00T N1 预训练骨干**、自采 **AirVLN-S + OpenFly** 约 3 万条 UE 轨迹，以及冻结 VLM + LoRA 微调 DiT；无公开权重/数据时无法按官方入口复现。
- **互指：** 升格实体页 [`wiki/entities/paper-fsd-vln.md`](../../wiki/entities/paper-fsd-vln.md)。未建 `sources/sites/` / `sources/repos/`。

## 摘要级要点

- **问题：** 空中 VLN 要同时做 **全局多模态推理** 与 **低延迟飞控**；现有反应式逐步预测易抖，自回归大模型又引入决策延迟。
- **架构：** 慢系统用预训练 VLM 把 RGB + 指令写成语义先验，写入 **VLSF** 缓冲；快系统用 **DiT**（交替 self-attn / cross-attn）在 UAV 状态、历史动作与缓存语义上生成动作；两路 **异步并行**。
- **动作：** 沿 AerialVLN / OpenFly / CityNav 的 **8 类离散原语**（前飞 3/6/9 m、左右转 30°、升降 3 m、Stop）；扩散在连续嵌入空间，再按阈值映回离散。
- **训练：** 数据集级自适应归一化 + **TW-MSE**（后段时间步权重大）；初始化 **GR00T N1**，冻结视觉–语言编码器，只训 DiT / 状态编码器 / 动作解码器；LoRA α=16，AdamW，200K step，4×RTX 4090，约 320 GPU-h。
- **数据：** AirVLN-S + OpenFly，四座虚拟城 + 广州渲染，**>30,000** 轨迹；长度集中 50–150 m，合并连续「前飞 3 m」后动作数约 20–50。
- **评测：** 成功半径 **20 m**。未见：SR **13.6%** / SPL **10.7** / NE **78**（自复现 OpenFly 5.1% / 3.5 / 198；CityNavAgent SR 11.7% 但 NE 60、OSR 更高）。已见：SR **26.7%** / SPL **22.8**。
- **延迟：** 单步动作 402→**176 ms**；214 条任务总时长 307.6→**144.7 s**（约 −53%）；双方都成功的 10 条仍约 −56%。
- **消融：** TW-MSE 比 MSE 更稳；动作视界 **H=1 最好**（SR 20.13%），H=2/4 下降——长程任务不等于长 chunk。
- **边界：** 仅低空仿真；无真机；摘要「未见 SR 最高 2×」主要相对 **OpenFly**，不是相对最强基线 CityNavAgent。

## 核心论文摘录（MVP）

### 1) 任务：空中长程 VLN 的推理–延迟结构冲突

- **链接：** <https://arxiv.org/abs/2607.08359> §1、§3.1
- **摘录要点：** 把长程空中 VLN 写成 \(p(a_{1:T}\mid I_{1:T},L)\)。全局语义推理吃大模型算力，飞行稳定要求非阻塞低延迟；单模型同时扛两件事会抖轨迹或拖决策。作者主张显式拆成快慢两路，而不是再堆一个端到端自回归控制器。
- **对 wiki 的映射：**
  - [FSD-VLN](../../wiki/entities/paper-fsd-vln.md)
  - [视觉–语言导航（VLN）](../../wiki/tasks/vision-language-navigation.md)
  - [具身大模型实时性 ↔ 泛化取舍](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md)

### 2) 方法：VLSF 缓冲 + DiT 快路 + 离散飞行动作

- **链接：** §3.2；Algorithm 1；Fig. 1–2
- **摘录要点：** 慢路 \(z_t=f_{\mathrm{VLM}}(I_t,L)\) 写入 VLSF；新 RGB 到达（UAV 到航点）才刷新。快路 \(a_{t:t+H}\sim p_\theta(a_{t:t+H}\mid z_t,s_t,a_{t-1})\)，DiT 对状态做 self-attn、对语义做 cross-attn。到达航点后更新观测，形成感知–动作环。
- **对 wiki 的映射：**
  - [FSD-VLN](../../wiki/entities/paper-fsd-vln.md)
  - [GR00T N1](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — 预训练骨干与双系统原型
  - [VLA](../../wiki/methods/vla.md)

### 3) 训练：自适应归一化 + TW-MSE

- **链接：** §3.3；§4.4；Fig. 5
- **摘录要点：** 按维做 \(\tilde x_d=(x_d-\mu_d)/\sigma_d\)，分位裁剪长尾。TW-MSE 对后段时间步加大权重，抑制长序列梯度振荡。作者报告 TW-MSE 收敛更平滑、最终损失更低，并改善导航指标。
- **对 wiki 的映射：**
  - [FSD-VLN](../../wiki/entities/paper-fsd-vln.md)

### 4) 评测：仿真 SR/SPL/延迟，以及 H 与 2× 口径

- **链接：** §4.1–4.4；Table 1–4
- **摘录要点：** 未见相对自复现 OpenFly：SR 5.1%→13.6%、SPL 3.5→10.7、NE 198→78。CityNavAgent 未见 SR 11.7%、NE 60、OSR 35.2，FSD-VLN 赢 SR/SPL、输 NE/OSR。H=1 优于 H=2/4。延迟与总时长约减半。无真机。
- **对 wiki 的映射：**
  - [FSD-VLN](../../wiki/entities/paper-fsd-vln.md)
  - [WorldVLN](../../wiki/entities/paper-worldvln-aerial-vln-wam.md) — 同为空中 VLN，但是 WAM + 真机
  - [Uni-LaViRA](../../wiki/entities/paper-uni-lavira.md) — 零样本 Aerial-VLN 对照

## 对 wiki 的映射（汇总）

- [`wiki/entities/paper-fsd-vln.md`](../../wiki/entities/paper-fsd-vln.md) — 主实体页
- [`wiki/tasks/vision-language-navigation.md`](../../wiki/tasks/vision-language-navigation.md) — 空中 / UAV 子域
- [`wiki/entities/paper-worldvln-aerial-vln-wam.md`](../../wiki/entities/paper-worldvln-aerial-vln-wam.md) — 空中 VLN · WAM 对照
- [`wiki/entities/paper-uni-lavira.md`](../../wiki/entities/paper-uni-lavira.md) — 零样本 Aerial-VLN 对照
- [`wiki/entities/paper-da-nav.md`](../../wiki/entities/paper-da-nav.md) — 地面城市 VLN 对照
- [`wiki/concepts/embodied-fm-latency-generalization-tradeoff.md`](../../wiki/concepts/embodied-fm-latency-generalization-tradeoff.md) — 异步双频实例
- [`wiki/methods/vla.md`](../../wiki/methods/vla.md) — GR00T 系导航适配
- [`wiki/entities/paper-hrl-stack-34-gr00t_n1.md`](../../wiki/entities/paper-hrl-stack-34-gr00t_n1.md) — 骨干来源
- [`wiki/overview/vln-open-source-repro-paradigms.md`](../../wiki/overview/vln-open-source-repro-paradigms.md) — 暂不可复现对照
- [`wiki/overview/multirotor-simulation-planning-control-stack.md`](../../wiki/overview/multirotor-simulation-planning-control-stack.md) — 多旋翼仿真栈交叉

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2607.08359>
