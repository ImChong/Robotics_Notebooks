# DR-VLA: Learning Human-Like Quadruped Navigation from Egocentric Web Videos via Domain-Invariant Scene Representations（arXiv:2610.18373）

> 来源归档（paper / arXiv v1）

- **标题：** DR-VLA: Learning Human-Like Quadruped Navigation from Egocentric Web Videos via Domain-Invariant Scene Representations
- **arXiv：** <https://arxiv.org/abs/2610.18373>（v1，2026-10-13；PDF：<https://arxiv.org/pdf/2610.18373>）
- **全文：** <https://arxiv.org/html/2610.18373>
- **项目页：** <https://developer.deeprobotics.cn/research/articles/dr-vla/index.html>
- **作者：** Jun Tian, Yingjie Niu*†, Ruo Chen, Hongyi Lin, Chao Li*
- **机构：** 云深处科技（DEEP Robotics）；伊利诺伊大学厄巴纳-香槟分校（UIUC）；浙江大学；清华大学；麻省理工学院（MIT）
- **入库日期：** 2026-11-21
- **核对范围：** 通读 `https://arxiv.org/html/2610.18373`（v1，2026-10-13）全文：Abstract、§1–§2、§3 任务建模、§4 方法（4.1 DISR 编码器、4.2 Mixture-of-Transformers 三专家架构与分块注意力掩码、4.3 EgoLynx 与两阶段训练）、§5 实验（5.1–5.5，Table 1–4 含 NAVSIM v1）、§6 结论与局限。边界：未复现训练；代码与数据集标注 Coming soon。
- **通读：** 已完成
- **一句话说明：** 通过固定语义词表的概念分割（SAM 3）+ 相对深度（Depth Anything 3）+ 因果时空自注意力构建跨域共享的域不变场景表示（DISR），并采用理解 / 世界 / 流匹配动作三专家 Mixture-of-Transformers，在 1 万小时无动作网络视频（EgoLynx-Web）上预训练后微调至 2,000 条真机配送遥操片段（EgoLynx-Robot），在留出测试集取得 QNS 0.884（超越 NaVILA 0.843）。

## 核心方法与实验要点（原文核对）

- **DISR（§4.1）：** 对每帧用冻结 SAM 3 按固定语义词表抽取 $|\mathcal{C}_{\mathrm{sem}}|$ 通道掩码 $S_i$，用 Depth Anything 3 抽取相对深度 $D_i$（舍弃绝对尺度以削弱相机高度差；均直接跑在未矫正鱼眼图上）。$E_\phi$ 对连续 $M=10$ 帧做因果时空自注意力得到紧凑潜变量 $\mathbf{z}_i \in \mathbb{R}^{N_z \times d_z}$，$D_\phi$ 重建 $(S_i, D_i)$。$E_\phi$ 训练后冻结，在标准 DR-VLA 中仅用作世界专家训练目标，推理期**无需**运行 SAM 3 / Depth Anything 3 / $E_\phi$。
- **Mixture-of-Transformers 三专家（§4.2）：** ① 理解专家 $f_u$（Qwen3-VL-2B）：多视角图像加视角嵌入 $e_v$ 与指令 $\ell$，训练期辅以场景描述自回归损失 $\mathcal{L}_{\mathrm{cap}}$；② 世界专家 $f_w$：可学习 world tokens + $L_w$ 层时空关系层，预测未来 $K$ 帧 DISR $\hat{\mathbf{z}}$（smooth-L1 + 余弦损失 $\mathcal{L}_{\mathrm{world}}$）；③ 动作专家 $f_a$：读取 $\mathbf{h}^u$、$\mathbf{h}^w$ 与自身状态 $s_t$，以条件流匹配经 4 步 Euler 积分输出未来 1 s 的 $N_a=10$ 个平面位姿 $(x, y, \theta)$。采用分块注意力掩码使动作 token 不可见于前缀与世界 token，推理期 $\mathbf{h}^u, \mathbf{h}^w$ 仅算一次并在 4 步积分中复用。
- **两阶段训练与数据（§4.3, §5.1）：** 在 32×H800 GPU 上训练。阶段 1 在 **EgoLynx-Web**（10,000 小时第一视角网络视频，无动作标签）训练理解与世界专家（20k 步，$\mathcal{L}_{\mathrm{cap}} + \lambda_w \mathcal{L}_{\mathrm{world}}$）；阶段 2 在 **EgoLynx-Robot**（2,000 条真实配送遥操片段，约 17 小时）全参微调并加入动作专家（5k 步）。
- **四足导航评测（§5.2–5.4, Table 1–2）：** EgoLynx-Robot 留出测试集上，DR-VLA 取得 QNS **0.884**（较最强基线 NaVILA 0.843 提高约 4.9%），FHE **0.041**（较 NaVILA 0.048 降 **14.6%**）；去掉 EgoLynx-Web 预训练则 QNS 降至 0.823。在预测目标消融（Table 2a）中，DISR（0.884）优于 Cosmos latent（0.845）、DINOv3（0.856）、V-JEPA 2.1 等视觉目标，且接近依赖激光点云与 3D 框/线标注的 OCC+BEV 目标（0.906），同时避免了在线 3D 感知的高时延。
- **跨领域泛化（§5.5, Table 4）：** 在自动驾驶 NAVSIM v1 `navtest` 上**不使用网络视频预训练**另训一个 64 候选提议-选择变体，纯相机输入达成 **92.1 PDMS**（去掉世界专家降为 89.5）。
- **原文局限（§6）：** ① DISR 定义在图像平面，仍存留视角差，需机器人数据微调而非零样本跨本体迁移；② 依赖现成视觉基础模型在固定词表下的伪标签，受长尾噪声与词表覆盖限制；③ 四足评测主要基于开环与伪闭环指标（QNS）加真机回放，大规模真机闭环部署留待后续。

## 对 wiki 的映射

- [DR-VLA（arXiv:2610.18373）](../../wiki/entities/paper-dr-vla.md)
