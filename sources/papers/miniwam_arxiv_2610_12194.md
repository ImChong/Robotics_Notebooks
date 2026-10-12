# MiniWAM 论文来源归档

## 元数据

- **题目：** MiniWAM: Learning Compact Future Targets for Efficient World-Action Modeling
- **方法：** MiniWAM + PRISM（Predictive Representations via Inverse Spatiotemporal Modeling）
- **作者：** Jie Chen、Ruofei Bai、Yuxin Cai、Yifeng Zhang、Chengyang He、Jun Li、Wei-Yun Yau、Guillaume Sartoretti
- **单位：** 新加坡国立大学（NUS）；南洋理工大学（NTU）；A*STAR 先进人工智能与计算研究所（据项目页）
- **预印本：** https://arxiv.org/abs/2610.12194
- **HTML：** https://arxiv.org/html/2610.12194v1
- **版本：** v1，2026-10-08，cs.RO
- **项目页：** https://j1dan.github.io/MiniWAM （归档见 [sources/sites/miniwam.md](../sites/miniwam.md)）
- **代码：** 截至 2026-10-12 未发布；项目页写 "Coming soon before Nov 2026"，论文称将发布实现、配置、检查点与评测脚本

## 论文要解决的问题

多数 WAM 联合预测动作与未来状态，但未来状态位于预训练视觉骨干（如 DINOv3、Wan2.1 VAE）的高维原生特征空间，大量训练算力花在与控制关系不大的视觉细节上。论文主张「有效的世界–动作建模不需要预测原生视觉未来」。

## 方法与训练信息

1. **两阶段。** Stage 1 学 PRISM 编码器；Stage 2 冻结 PRISM，训练 MiniWAM 联合预测紧凑未来表示和动作。
2. **PRISM 编码器：** 轻量 3D 卷积网络（卷积–GroupNorm–SiLU）。VAE 路径三块，通道 (32,64,128)，特征网格 (5,28,28)→(3,4,4)；DINOv3 路径通道 (64,128,128)，网格 (5,14,14)→(3,4,4)；1×1×1 头得到 D_L=64；对角高斯瓶颈，Stage 2 取后验均值作确定性目标。每相机 3×4×4×64，即 48 个「世界 token」。
3. **Stage-1 损失：** λ_a·流匹配动作损失（逆动力学：潜变量干净、动作加噪）+ λ_rec·特征重建 MSE（λ_rec=1）+ β·KL（β=1e-6，1,000 次更新线性 warmup）；解码器在 Stage 1 后丢弃。
4. **冻结骨干特征：** Wan2.1 VAE（224×224，一当前 + 四个 16×28×28 未来块，16 帧窗口）；DINOv3 ViT-B/16（LVD-1689M，patch token 17×196×768，1×1 卷积降到 64）。
5. **MiniWAM 结构：** 约 247.5M 策略参数（0.25B）；12 层 Mixture-of-Transformers，世界与动作两个 DiT 专家，宽度 768、FFN 3072、12 头；动作为 flow matching；非 VLM，视觉骨干与 UMT5 文本编码器为外部冻结。Stage 1 中动作可看潜变量但反向不可；Stage 2 世界–动作双向注意力；语言与本体感觉走 cross-attention。推理 20 步去噪，LIBERO 每 10 步重规划、动作块 16，RoboTwin 每 24 步、动作块 24。
6. **训练：** LIBERO Stage 1 为 15 epoch（16,260 次更新）、Stage 2 为 50 epoch（54,200 次）；RoboTwin 分别 6 / 25 epoch；AdamW，峰值学习率 5e-4，余弦调度，5% warmup，全局 batch 256，BF16。效率计时在一张 RTX PRO 6000（96GB）；主训练 GPU 数原文未报告。
7. **数据：** LIBERO 四套件 40 任务（评测 2,000 episode）；LIBERO-Plus 全集 10,030 个扰动任务（消融用 2,000 任务子集 LIBERO-Plus-2K）；RoboTwin 2.0 共 50 双臂任务、每任务 50 条示教、三相机；LIBERO-90 仅用于表示分析。

## 论文报告的结果

- **65× 的来源：** 原生未来网格 3,136 个位置（VAE 4×28×28；DINOv3 16×14×14）对 PRISM 的 48 个位置 ≈ 65.3×；这是网格位置数而非序列长度，Transformer token 数仅 784→48（约 16.3×）。
- **主表（LIBERO / LIBERO-Plus / RoboTwin 2.0 成功率）：** MiniWAM 0.25B：97.2 / 73.6 / 46.1。对照：π0.5(3.3B) 96.4 / 85.7 / 58.4；π0(3.3B) 93.5 / 70.5 / 31.4；OpenVLA-OFT(7B) 96.6 / 71.4 / –；Fast-WAM(6B) 95.8 / 71.0 / 39.9；DreamWAM(6B) 96.9 / 75.4 / –；AHA-WAM(7.2B) –/–/33.8；Xiaomi-Robotics-0(4.7B) 98.7 / – / 40.5。RoboTwin 使用 clean 训练赛道（clean2clean 与 clean2random 均值）。
- **受控对比（Table 1，LIBERO / LIBERO-Plus-2K）：** VAE：无共训 62.3/25.1，原生共训 92.8/50.2，PRISM 94.8/56.5；DINOv3：无共训 89.0/56.5，原生共训 93.2/60.9，PRISM 97.2/74.1。
- **训练吞吐（一张 RTX PRO 6000，全局 batch 256）：** RoboTwin、VAE、horizon 32：原生 7.79 → PRISM 81.66 样本/秒，Stage 2 加速 10.48×，含一次性 Stage 1 成本的总加速 8.46×（即「最高 8×」）；LIBERO horizon 16 总加速 3.05–3.55×。
- **Stage-1 目标消融（DINOv3，LIBERO-Plus-2K）：** 4×4,D=64：仅重建 69.5，仅逆动力学 70.5，两者 74.1；瓶颈缩到 2×2 为 67.8，1×1(D=16) 为 61.7。
- **表示分析（LIBERO-90 未见任务）：** 跨骨干 CKA 0.372/0.249→0.795/0.705；留一任务 Jaccard@5 0.170/0.107→0.512/0.333；动作 RMSE 0.400/0.370→0.365/0.272（主视角/腕部视角）。
- 无真机实验。

## 复现与解释边界

- PRISM 针对每个下游设置单独学习；两阶段设计冻结 PRISM；表示分析为单检查点、单随机种子。
- 论文未明确主表 LIBERO-Plus 数字用全集还是 2K 子集；RoboTwin 对比为 clean 训练赛道，与部分方法 clean+randomized 训练的报告不可直接比较。
- 效率数字是单卡吞吐估计，「65×」不是 wall-clock 或序列长度比。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-miniwam-compact-future-targets.md
- 概念入口：wiki/concepts/world-action-models.md

## 一手来源

- arXiv 摘要：<https://arxiv.org/abs/2610.12194>
- arXiv HTML：<https://arxiv.org/html/2610.12194v1>
- 项目页：<https://j1dan.github.io/MiniWAM>
