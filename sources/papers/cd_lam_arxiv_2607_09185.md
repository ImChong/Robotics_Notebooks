# Causally Debiased Latent Action Model for Embodied Action-Conditioned World Models（arXiv:2607.09185）

> 来源归档（ingest）

- **标题：** Causally Debiased Latent Action Model for Embodied Action-Conditioned World Models
- **短名：** CD-LAM
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2607.09185>（v1 2026-07-10；**v2 2026-09-26**；cs.CV / cs.RO）
- **PDF：** <https://arxiv.org/pdf/2607.09185>
- **作者：** Yufan Wei、Kun Zhou、Lingjun Mao、Ziming Xu、Shuang Liang、Zijun Zhang、Ziqiao Xi、Ruobing Han、Yuchen Yan、Xinyue Wang、Fan Feng、Biwei Huang（12 人）
- **机构：** 以太智能（Aether AI）、加州大学圣地亚哥分校（UCSD）（arXiv v2 HTML 署名；含「Done during an internship at Aether AI」标注）
- **项目页：** <https://yufanwei.github.io/CD-LAM-project-page/>
- **代码：** <https://github.com/AetherLabsAI/CD-LAM>（v2 元数据）/ <https://github.com/yufanwei/CD-LAM>（博客与 README），Apache-2.0
- **权重：** <https://huggingface.co/AetherLabs-AI/CD-LAM> / <https://huggingface.co/yufanwei/CD-LAM>（2B：LAM、pretrain、posttrain）
- **入库日期：** 2026-10-10
- **博客归档：** [aether_cd_lam.md](../blogs/aether_cd_lam.md)
- **一句话说明：** LAM 的纯重建目标把动作相关动力学与动作无关视觉混杂缠在一起，导致下游 ACWM 零动作仍有残余运动、换动作也不跟；CD-LAM 用具身中心重建 + 动作中心对比 + 潜空间校准三项目标去偏，DreamDojo-2B/14B、LTX-2.3-22B、X-VLA 上均有收益。

## 开源状态（步骤 2.5，2026-10-10）

- **结论：已开源（代码 + 2B 权重）**；14B 权重与 14B 适配器未包含。
- README 运行路径：`bash setup.sh --accept-base-license [--with-models]` → `bash run.sh prepare-agibot ...`（AgiBotWorld Alpha）/ `scripts/download_datasets.py egodex` → `bash run.sh runtime-doctor --stage all` → `bash run.sh pipeline`（或 `stage1` / `bridge` / `stage2` / `stage3` 分步）→ `bash run.sh score-fdce --tracks ... --output evaluation/fdce.json`。
- 环境要求：Linux x86-64、Python 3.10、PyTorch 2.7.0+cu128、Ampere / Hopper GPU、约 30 GB 空间。
- 接口：LAM 输出 **32 维** 潜动作（不需要 bridge）；Stage 3 / rollout 用 **22 维** 机器人动作，需要 checkpoint 专属 bridge。

## 核心摘录（v2 摘要与正文，面向 wiki 编译）

- 摘要：ACWM 需要大量带动作标签数据；LAM 从无标签视频推潜动作缓解这一瓶颈，但纯重建目标使 LAM 偏向视觉上下文而非动作动力学。
- v2 主数字：潜动作 / 机器人动作条件下动作跟随误差分别最多降 **42% / 35%**；追平 DreamDojo 参考所需适配更新少 **>12×**（14B：FDCE 约 3k、PSNR 约 4k 步追平参考 50k 步）。
- **LTX-2.3-22B 适配**（v2 Table 5，自报）：

| 模型 | 潜动作预训练 | 机器人动作适配 | PSNR ↑ | FDCE 均值 ↓ | FDCE 中位 ↓ |
|------|--------------|----------------|--------|-------------|-------------|
| DreamDojo 14B | 256×H100，140k 步，143.36M 样本 | 128×H100，50k 步，25.6M 样本 | 20.01 | **11.11** | 8.98 |
| CD-LAM + LTX-2.3 22B | 8×H200，60k 步，3.84M 样本 | 7×H200，28k 步，1.568M 样本 | **20.38** | 11.75 | **7.82** |

  - 合计样本暴露约为 DreamDojo-14B 的 **3.2%**（潜动作阶段少 37.3×，适配阶段少 16.3×）；FDCE 均值仍略差于 DreamDojo-14B。
- **X-VLA 策略预训练**（v2 Table 6，成功率 %，自报）：

| 预训练 | LIBERO | LIBERO-Plus | RoboTwin C2R Clean | RoboTwin C2R Random |
|--------|--------|-------------|--------------------|---------------------|
| X-VLA（原始） | **98.05** | 69.74 | 85.67 | 25.00 |
| DreamDojo LAM | 95.72 | 75.69 | 84.00 | 28.83 |
| CD-LAM | 96.30 | **78.25** | **86.67** | **33.83** |
| Δ vs DreamDojo LAM（pp） | +0.58 | +2.56 | +2.67 | +5.00 |

  - 注意：LIBERO 上 CD-LAM（96.30）仍低于不做潜动作预训练的 X-VLA（98.05）；摘要的「最多 +5pp」是相对 DreamDojo LAM 而言。
- 评测数据：潜动作 rollout 在 EgoDex 上评，机器人动作 rollout 在留出的 AgiBot 片段上评。

**对 wiki 的映射：** [paper-cd-lam](../../wiki/entities/paper-cd-lam.md)
