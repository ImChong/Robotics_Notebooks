# RT-2 论文通读（VLA 定义与训练配方视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/vla.md`）

- **标题：** RT-2: Vision-Language-Action Models Transfer Web Knowledge to Robotic Control
- **作者：** Brohan, Brown, Carbajal 等（Google DeepMind）
- **类型：** paper / vla
- **原文：** <https://arxiv.org/abs/2307.15818>（2026-11 通过 arXiv PDF 获取全文）
- **一句话说明：** 首次把「VLA」作为一类模型提出：把动作表示为文本 token，与互联网视觉语言数据一起 co-fine-tune 现成 VLM，使机器人策略继承网络语义知识。
- **核对范围：** 通读摘要、§1 引言、§2 相关工作、§3 模型与训练（3.1–3.3）、§4 实验（4.1–4.4）、§5 局限、§6 结论；附录（数据集细节、基线细节、模型架构 D、训练细节 E、完整指令表与数值表 F/H、CoT 样例 I）仅浏览，未逐项核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **VLA 定义（摘要、§1）：** 把动作当作文本 token，与自然语言 token 同等地放进模型训练集；这类模型称 vision-language-action（VLA），RT-2 是其一例；沿用 RT-1 的数据协议，但把骨干换成大型 VLM，不引入新的动作专用参数。
- **骨干（§3.1、§4）：** 适配 PaLI-X（5B、55B）与 PaLM-E（12B），分别记 RT-2-PaLI-X / RT-2-PaLM-E。
- **动作表示（§3.2）：** 动作为末端 6-DoF 位置/旋转位移 + 夹爪开合 + 终止指令；连续维度均匀离散为 256 bin，共 8 个整数；PaLI-X 直接用整数 token，PaLM-E 覆盖 256 个最少使用的 token（symbol tuning）；输出串形如「terminate Δpos×3 Δrot×3 gripper」；输入为 VQA 格式 prompt「Q: what action should the robot take to [task]? A:」。
- **Co-fine-tuning（§3.2）：** 同时训练机器人数据与原始网络数据，并在 batch 中提高机器人数据采样权重；比只用机器人数据微调泛化更好。
- **输出约束（§3.2）：** 机器人任务下解码只采样合法动作 token。
- **实时推理（§3.3）：** 55B 模型部署在多 TPU 云服务上，经网络查询；55B 约 1–3 Hz，5B 约 5 Hz；作者称其为当时用于闭环控制的最大模型（超出一个数量级以上）。
- **数据（§4）：** 网络数据（VQA、captioning 等）+ RT-1 的机器人示教（13 台机器人、17 个月、办公厨房场景）。
- **结果（§4.1）：** 已见任务上与 RT-1 相近；未见物体/背景/环境泛化上相对 RT-1、MOO 约 2 倍，相对其余基线约 6 倍。Language-Table 仿真：RT-2-PaLI-3B 90±10，高于 BC-Zero 72、RT-1 74、LAVA 77。
- **涌现能力（§4.2）：** 符号理解、推理（视觉/数学/多语言）、人物识别三类；RT-2-PaLI-X 平均成功率超过次优基线 RT-1 的 3 倍。
- **消融（§4.3）：** 从零训练 5B 模型表现很差；co-fine-tune 优于仅机器人数据微调；模型越大泛化越好。
- **思维链（§4.4）：** 数据中加入自然语言「Plan」步骤（先 Plan 后 Action），仅需数百步微调；定性展示可处理「找个当锤子的东西」之类多步语义推理；仅为定性证据。
- **局限（§5）：** (1) 不会获得新的运动技能，物理技能仍受限于机器人数据中的技能分布；(2) 推理计算代价高，高频控制可能成为瓶颈，需量化/蒸馏；(3) 可用于构造 VLA 的开放 VLM 很少。

## 对 wiki 的映射

- [`wiki/methods/vla.md`](../../wiki/methods/vla.md)
- [`wiki/entities/paper-rt-2.md`](../../wiki/entities/paper-rt-2.md)
