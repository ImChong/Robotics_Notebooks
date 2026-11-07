# PaliGemma: A versatile 3B VLM for transfer（arXiv:2407.07726）

> 来源归档（一手论文）

- **标题：** PaliGemma: A versatile 3B VLM for transfer
- **类型：** paper / VLM / 视觉-语言基座模型
- **arXiv：** <https://arxiv.org/abs/2407.07726>（PDF：<https://arxiv.org/pdf/2407.07726>）
- **机构：** Google DeepMind（Lucas Beyer、Andreas Steiner、André Susano Pinto、Alexander Kolesnikov、Xiao Wang、Xiaohua Zhai 等）
- **版本：** arXiv v2（2024-10-10）；报告日期 2024-07
- **入库日期：** 2026-11-07
- **核对范围：** 2026-11-07 通读 PDF 正文：摘要与相关工作、§3 模型（架构、Stage0–3、预训练任务混合、其他训练细节）、§4 结果（Table 1）、§5 全部消融（预训练时长、掩码与目标、新 token 初始化、冻结与否、连接器、有无图像编码器、分辨率）、§6 可迁移性、§7 要点、§8 结论；附录 A（更多相关工作）、C–H（RefCOCO 增强、CountBenchQA、WidgetCaps 数字问题、框标注 vs 提示、Objaverse、多任务迁移 Table 4）、I（推理耗时 Table 5）、J（超参表头）；附录 B 的逐任务样例与 K 的逐任务消融图只浏览；参考文献与作者贡献表未读；未运行 big_vision 代码或权重，也不核对 Gemma / SigLIP 原论文。
- **通读：** 已完成
- **一句话说明：** SigLIP-So400m 图像编码器 + Gemma-2B 解码器 + 线性连接层的 3B 开放基座 VLM；Stage0–3 训练，Stage1 约 10 亿样本、全参数不冻结；224/448/896 三个分辨率检查点；在近 40 个任务上迁移验证。

## 核心摘录（面向 wiki 编译）

1. 架构：SigLIP So400m（形状优化 ViT）→ 零初始化线性投影 → Gemma-2B v1.0；token 序列 = 图像 token + BOS + prefix + SEP + suffix + EOS；图像与 prefix 全注意力（prefix-LM），suffix 自回归掩码；输入图像固定缩放为 224/448/896 方形，分别对应 256/1024/4096 个图像 token。
2. 训练：Stage0 复用现成单模态检查点；Stage1 224px、序列 128、约 10 亿样本，全模型不冻结（图像编码器学习率慢热身）；Stage2 依次到 448px（+5000 万样本）与 896px（+1000 万样本），序列长度 512；Stage3 迁移（全参数微调）。预训练用 TPUv5e-256，Stage1 不到 3 天，每个 Stage2 约 15 小时，MFU 55%。
3. 新增词表：1024 个位置 token `<loc0000>`–`<loc1023>` 与 128 个 VQ-VAE 分割 token `<seg000>`–`<seg127>`；标准高斯小噪声初始化优于平均嵌入初始化。
4. 消融结论：Stage1 越长越好、跳过最差（100M 样本是消融的折中点）；prefix-LM + 仅 suffix 监督最佳；图像编码器与语言模型都不冻结更好；线性连接器不逊于 MLP；无图像编码器（Fuyu 式）可行但样本效率明显更低；分辨率收益约一半来自信息量、一半来自序列长度，需要分辨率专属检查点。
5. 结果（Table 1，224px / 448px / 896px）：COCOcap 141.9 / 144.6；VQAv2 83.2 / 85.6；TextVQA 55.5 / 73.2 / 76.5；DocVQA 43.7 / 78.0 / 84.8；RefCOCO testA 75.7 / 77.9 / 78.7；MMVP 47.3（论文称 GPT-4V 38.7、Gemini 40.7）。
6. 可迁移性：5 次重跑多数任务标准差 <0.5；单一推荐超参相对逐任务搜索的相对遗憾 <2.5% 的任务有 37 个；用 4k 迁移样本多数任务可达全量成绩的 90%、256 样本约 80%。

## 对 wiki 的映射

- 沉淀实体页：[`wiki/entities/paper-rcl-2407-07726-paligemma-a-versatile-3b-vlm-for-transfer.md`](../../wiki/entities/paper-rcl-2407-07726-paligemma-a-versatile-3b-vlm-for-transfer.md)
- 机器人 VLA 下游：[π₀](../../wiki/entities/paper-pi0.md)（以 PaliGemma 为 VLM 骨干；以 π₀ 论文为准，本文不含机器人内容）

## 参考来源（原始）

- arXiv：<https://arxiv.org/abs/2407.07726>
- 代码（论文声明训练基于 big_vision）：<https://github.com/google-research/big_vision>
