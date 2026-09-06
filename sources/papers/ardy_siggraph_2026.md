# ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation

> 来源归档（ingest）

- **标题：** ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation
- **类型：** paper
- **来源：** ACM Transactions on Graphics（SIGGRAPH 2026）
- **arXiv：** <https://arxiv.org/abs/2607.08741>（2607.08741）
- **原始链接：**
  - 项目页：<https://research.nvidia.com/labs/sil/projects/ardy/>
  - PDF：<https://research.nvidia.com/labs/sil/projects/ardy/assets/ardy_paper.pdf>
  - DOI：<https://doi.org/10.1145/3811284>
  - 代码：<https://github.com/nv-tlabs/ardy>
  - 模型：<https://huggingface.co/collections/nvidia/ardy>
- **机构：** NVIDIA Research · ETH Zürich
- **入库日期：** 2026-07-11（深度更新 2026-09-06）
- **一句话说明：** **自回归扩散** 交互式人体运动：混合 **显式 root + 潜空间 body** 与 **两阶段 Transformer 去噪**；**33 ms** 级 4-step 推理；支持 **在线文本 + 长时域运动学约束**；HumanML3D 与 Bones Rigplay 双轨评测。

## 核心论文摘录

### 1) 动机：离线可控 vs 在线实时

- **离线**（Kimodo 等）：文本+运动学约束强，但 **推理太慢** 无法交互。
- **在线**（MotionStreamer、DiP 等）：实时但常 **缺约束**、**短上下文** 或需 test-time optimization / RL control。
- **ARDY 目标：** 流式生成 + **在线 prompt** + **超出当前窗口的远期路点/关键帧** + **原生** 约束（无额外控制模块）。

### 2) 方法要点

- **Hybrid representation：** patch 化 body → latent embedding + **显式 global root**（Motion Tokenizer 编解码）。
- **Two-stage AR denoiser：** 可变历史上下文；窗口内预测 C 个干净 token；**denoise 循环内先 root 后 body**（交错两阶段）。
- **约束：** mask 化运动序列注入；时间/关节可稀疏；可落在 **generation window 之外**（history 8s / future 10s，Table 1）。

### 3) 训练数据

- **主训练 / Demo / 消融：** **Bones Rigplay** ~700h 工作室级动捕 + 文本（150+ 演员；27-joint unified skeleton）。
- **HumanML3D 对比实验：** ~30h 公开数据；保留 SMPL 关节旋转（非原版 IK 后处理）。

### 4) 关键数字（论文）

| 项 | 数值 |
|----|------|
| 交互延迟（4-step，RTX 4090，G=40@20fps） | **~33 ms**（10-step ~63 ms） |
| HumanML3D 自回归对比（Table 5） | 相对 DiP / DartControl 等：**更低 FID、约束误差与 latency**（详见论文 Table 4–5） |
| Rigplay 消融 | 8-frame horizon **更快 prompt 切换**；40-frame **更高 FID/R-prec** |

### 5) 下游：ARDY + SONIC → G1

- 实时人形运动 + **GEAR SONIC** 物理跟踪 → **Unitree G1** 交互（芭蕾等演示）。

## 对 wiki 的映射

- [ARDY 实体页](../../wiki/entities/ardy.md)
- [Kimodo](../../wiki/entities/kimodo.md) — 离线姊妹
- [SONIC](../../wiki/methods/sonic-motion-tracking.md) — 跟踪下游

## 引用（BibTeX）

```bibtex
@article{zhao2026ardy,
  title   = {ARDY: Autoregressive Diffusion with Hybrid Representation for Interactive Human Motion Generation},
  author  = {Zhao, Kaifeng and Petrovich, Mathis and Zhang, Haotian and Wang, Tingwu and Tang, Siyu and Rempe, Davis},
  journal = {ACM Transactions on Graphics (TOG)},
  year    = {2026},
  volume  = {45},
  number  = {4},
  articleno = {86},
  doi     = {10.1145/3811284}
}
```
