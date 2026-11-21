# X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining（arXiv:2606.14752）

> 来源归档（paper）

- **标题：** X-Tokenizer: A Multimodal Action Tokenizer for Vision-Language-Action Pretraining
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2606.14752>（v1 2026-06-07；v2 2026-06-28，cs.CV）
- **HTML：** <https://arxiv.org/html/2606.14752v2>
- **PDF：** <https://arxiv.org/pdf/2606.14752>
- **项目页：** <https://x2robot.com/pages/x-tokenizer>；镜像 <https://x-square-robot.github.io/X-Tokenizer_projectPage/>
- **代码：** <https://github.com/X-Square-Robot/X-Tokenizer>（Apache-2.0，推理侧）
- **权重：** <https://huggingface.co/x-square-robot/X-Tokenizer>（`xtokenizer.pth`）
- **作者：** Miracle Kang, Lights Shi, Lucy Liang, Roy Gan, Dongxiu Liu, Pushi Zhang, Sylas Chen, Shawn Qin, Yinan Zheng, Jinliang Zheng, Hao Wang, Xianyuan Zhan, Hang Su（X Square Robot、香港城市大学、清华大学）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2606.14752v2` 正文，含摘要、§1 引言、§2 相关工作、§3 方法（Encoder–SRQ–Decoder 与 MAM / VL Contrastive Alignment / Next-Frame VL Prediction 三头）、§4 实验（多模态对齐探针、SRQ 码本分析与 Table 1–2、RoboTwin 2.0 50 任务、真机 7 任务对比）及 §5 结论与未来工作；未运行真机训练与预训练
- **通读：** 已完成

## 核心摘录（通读 arXiv:2606.14752v2）

- **核心定位（§3.1）：** 将动作分词器从纯几何压缩转向「与消费端 VLM 对齐的语义接口」。训练后移除三个辅助头，仅保留轻量的 Encoder–SRQ–Decoder 为混合 VLA 离线编码离散监督信号；下游部署时只运行连续 flow 头，不调用 X-Tokenizer。
- **核心架构（§3.2–§3.3）：**
  - 26 维 delta 动作空间（相对 chunk 前一帧本体感觉锚点 $o$）；Perceiver 式编码器下采样 $4\times$（$T=64\to M=16$），经 4 级 $\times 2048$ 码本的 Semantic Residual Quantization（SRQ）量化，Perceiver IO 解码器还原绝对动作。
  - **非对称语义监督：** 第一级码本 $c^{(1)}_{1:M}$ 接受 Masked Action Modeling（MAM），量化前潜变量 $h_{1:M}$ 与冻结 Qwen2.5-VL-7B（三视角图像+指令）做槽位级+全局级 InfoNCE 对比对齐，量化潜变量 $\tilde z_{1:M}$ 做下一帧 VL 特征预测；第 2–4 级只承担重建与 commitment 损失。
- **关键实验（§4）：**
  - 预训练语料约 2.4M 轨迹 / 2.0B 动作帧（17 个机械臂族）。
  - 码本分工（§4.2.1–4.2.2）：激活率 76.4% / 93.8% / 99.3% / 99.8%，PPL 单调递增 $510\to 700\to 828\to 916$；重建 $\ell_1=0.01693$ 故意高于 FAST（0.01446）以换取语义结构；噪声 $\sigma=0.008$ 下 WER 为 0.526（FAST 1.445），编辑集中在 $q_{1:3}$ 而 $q_0$ 稳定。
  - RoboTwin 2.0（§4.3）：Wall-OSS+X-Tokenizer 均分 82.8（Easy 84.7 / Hard 80.9，vs $\pi_{0.5}$ 79.8）；5 单臂跨本体联合训练 Hard 从 64.0 升至 74.4。
  - 真机 7 任务受控对比（§4.4，Qwen2.5-VL-3B 骨干）：+X-Tokenizer 均分 77.4 PR / 85.9% VQA（vs +FAST 73.0 / 75.7%，+RVQ no-aux 69.1 / 79.4%）。
- **未来方向（§5）：** 扩展至灵巧手与关节空间控制（突破末端锚点限制），探索自适应语义–重建权衡调度。

## 对 wiki 的映射

- [cn-os-x-tokenizer](../../wiki/entities/cn-os-x-tokenizer.md)
- 代码归档：[x-tokenizer.md](../repos/x-tokenizer.md)
- 项目页归档：[x2robot-x-tokenizer.md](../sites/x2robot-x-tokenizer.md)
