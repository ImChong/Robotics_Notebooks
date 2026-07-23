# S²-VLA: Decoupling Semantic and Spatial Streams in Vision-Language-Action Models for Autonomous Driving（arXiv:2607.13926）

> 来源归档（ingest）

- **标题：** S²-VLA: Decoupling Semantic and Spatial Streams in Vision-Language-Action Models for Autonomous Driving（arXiv abs 元数据亦写作 *S-squared-VLA*）
- **缩写：** **S²-VLA** / **S-squared-VLA**（Semantic–Spatial Dual-Stream VLA；**勿与** 操作域 *S2-VLA: State-Space Guided…* arXiv:2606.27872 混淆）
- **类型：** paper / vla / autonomous-driving / end-to-end-planning / navsim
- **arXiv：** <https://arxiv.org/abs/2607.13926>（Submitted 2026-07-15；PDF：<https://arxiv.org/pdf/2607.13926>；HTML：<https://arxiv.org/html/2607.13926>）
- **项目页：** **无独立项目页**（截至 2026-07-23 仅 arXiv）
- **代码：** **未开源**（论文与 arXiv 页未列 GitHub / HF / 权重）
- **作者：** Jianguo Yu\*、Rukang Wang\*、Duanfeng Chu（通讯）、Chen Wang、Renju Feng、Liping Lu（\*共一）
- **机构：** 武汉理工大学机械与电子工程学院；智能交通系统研究中心；计算机科学与人工智能学院
- **入库日期：** 2026-07-23
- **一句话说明：** 针对驾驶 VLA 的 **spatial representation collapse**，把 **语义流**（InternVL3-2B 多尺度 + action queries）与 **空间流**（ViT 稠密特征 + BEV map / agent 辅助监督）解耦，再用 **Dual-Stream Planning Adapter** 级联融合，在 **NAVSIM** 闭环、纯 **SFT** 设定下报 **PDMS 87.1**、**NC 98.4**。

## 开源状态（步骤 2.5）

- **核查（2026-07-23）：** arXiv abs / HTML 全文无项目页、GitHub、Hugging Face 或权重下载链；结论段落亦未承诺「code will be released」。
- **结论：** **确认未开源**。wiki「工程实践 / 源码运行时序图」写明不适用。

## 摘录 1：问题与主张（§I）

- **语义–物理鸿沟：** VLM 输出离散自回归符号，车控需要连续轨迹；token 化与深层抽象会丢掉几何细节。
- **VLA 新瓶颈：** 单流把深层语义直接映射到控制 → **spatial representation collapse**（细粒度空间/几何先验不可逆退化）与 semantic-over-geometry 偏置。
- **主张：** **S²-VLA** = 语义流（层级桥接 / 多尺度）∥ 空间流（绕过语言瓶颈，保留未压缩视觉特征 + 感知辅助）→ Dual-Stream Planning Adapter。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-s-squared-vla.md`](../../wiki/entities/paper-s-squared-vla.md)；与 [VLA](../../wiki/methods/vla.md)、[X-Foresight](../../wiki/entities/paper-x-foresight.md)、[自动驾驶算法地图](../../wiki/overview/autonomous-driving-core-algorithms-series.md)、[Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md) 互链。

## 摘录 2：架构三件套（§II）

| 模块 | 要点 |
|------|------|
| **Multi-Scale Semantic Stream** | 骨干 **InternVL3-2B**（InternViT + Qwen2.5）；注入 \(N_{\mathrm{act}}=64\) action queries；稀疏采样层 \(L=\{3,8,13,18,23,24\}\)；ego 历史 MLP → \(E_{\mathrm{ego}}\) 拼入 action 记忆 |
| **Task-Driven Spatial Stream** | 动态分辨率 9-patch；每 patch \(N_{\mathrm{vis}}=64\) visual queries；绕过自回归 LM；**Map Head** → 局部 BEV 语义图（\(X\in[0,32]\) m，\(Y\in[-32,32]\) m）；**Agent Head** DETR 式 \(N_{\mathrm{agent}}=30\) |
| **Dual-Stream Planning Adapter** | \(M=8\) planning tokens；每块先语义/状态对齐（gated MHCA），再视觉空间精修（对 \(V_{\mathrm{spatial}}\) MHCA）；MLP 解码 \(\hat{Y}\in\mathbb{R}^{M\times 3}\) |
| **损失** | \(\mathcal{L}_{\mathrm{total}}=\lambda_{\mathrm{plan}}\mathcal{L}_{\mathrm{plan}}+\lambda_{\mathrm{agent}}\mathcal{L}_{\mathrm{agent}}+\lambda_{\mathrm{map}}\mathcal{L}_{\mathrm{map}}\)；规划含 \(L_1\) + 加速度/jerk 平滑 |

**训练（§III-B，三阶段）：** (1) ReCogDrive VQA 上 SFT VLM 3 epoch；(2) 冻 VLM，LoRA + 意图/辅助感知从头训 4 epoch；(3) 冻感知与意图，专训视觉精修模块 4 epoch。AdamW，4×A100，batch 16，LoRA \(r=8,\alpha=16\)。

**对 wiki 的映射：** 实体页画「语义∥空间 → Adapter → 轨迹」流程图；强调 **纯相机、无 LiDAR** 与 **SFT-only** 对照设定。

## 摘录 3：NAVSIM 结果与消融（§IV）

| 设定（navtest，SFT-only） | PDMS ↑ | NC ↑ | 备注 |
|---------------------------|--------|------|------|
| InternVL3-2B（文本头轨迹） | 84.1 | 97.6 | 同骨干朴素 VLM |
| ReCogDrive\* | 86.5 | 98.1 | 排除其 RL 变体 |
| ImagiDrive | 86.4 | 97.9 | |
| **S²-VLA** | **87.1** | **98.4** | 论文称纯 SFT VLA SOTA；NC 最高 |
| ARTEMIS（E2E，LiDAR） | 87.0 | 98.3 | 对照 |
| DiffusionDrive（E2E，LiDAR） | 88.1 | 98.2 | 更高 PDMS，但依赖扩散解码 + 点云 |

消融（Table III）：BaseVLM 84.1 → +语义 85.6 → +空间 86.2 → +辅助感知 **87.1**。

**局限（§V）：** 双流稠密融合算力开销；架构与训练范式正交，下一步可接闭环 RL（如 GRPO）后训练。

**对 wiki 的映射：** 用「驾驶 VLA 的空间保真 vs 语义推理」对照 [X-Foresight](../../wiki/entities/paper-x-foresight.md)（内嵌世界模型）与 [Qwen-RobotNav](../../wiki/entities/qwen-robot-nav.md)（NAVSIM PDMS 另一路线）。

## 建议 wiki 动作

- 新建 **`wiki/entities/paper-s-squared-vla.md`**（含流程总览；源码时序图标不适用）。
- 注册机构 **`whut`**（武汉理工大学）。
- 更新 **`wiki/methods/vla.md`**、**`wiki/entities/paper-x-foresight.md`**、**`wiki/overview/autonomous-driving-core-algorithms-series.md`**、**`wiki/entities/qwen-robot-nav.md`**、**`wiki/tasks/vision-language-navigation.md`** 交叉引用。
