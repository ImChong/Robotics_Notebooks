# Being-H0.7: A Latent World-Action Model from Egocentric Videos

> 来源归档（paper / arXiv:2605.00078v1）

- **URL：** <https://arxiv.org/abs/2605.00078>
- **PDF：** <https://arxiv.org/pdf/2605.00078.pdf>
- **项目页：** <https://research.beingbeyond.com/being-h07>
- **代码仓库：** <https://github.com/BeingBeyond/Being-H>（`Being-H07/README.md` @ `27ec67b`，截至 2026-11-21 仅含说明文档，未释出 H0.7 源码与权重）
- **标题：** Being-H0.7: A Latent World-Action Model from Egocentric Videos
- **作者：** Hao Luo, Wanpeng Zhang, Yicheng Feng, Sipeng Zheng, Haiweng Xu, Chaoyi Xu, Ziheng Xi, Yuhui Fu, Zongqing Lu（BeingBeyond Team）
- **机构：** BeingBeyond（智在无界）、Peking University、Renmin University of China、Tsinghua University
- **年份：** 2026（arXiv 2026-04-30）
- **核对范围：** 已获取并通读 `arXiv:2605.00078v1` PDF 抽取全文（Sections 1–5，涵盖 Latent Query + Prior/Posterior 双分支 MoT 打包、前瞻 V-JEPA2.1 Perceiver 嵌入、Frobenius 对齐损失与范数/谱熵防坍塌正则、6 项仿真与 12 项真机盲测及 UAC 延迟数据）及 GitHub `BeingBeyond/Being-H/Being-H07/README.md @ 27ec67b`；未独立复现训练与真机实验。
- **通读：** 已完成

## 摘要与核心贡献

1. **隐式世界-动作模型（Latent WAM）定位**：指出直接 VLA 易从稠密观测学到动作捷径，而显式像素生成式 WAM 计算开销大、且重建纹理光照与动作控制无关。Being-H0.7 在多模态上下文 $c = [x; o_{-H:0}; s]$ 与含噪动作块 $a_{0:T}$ 之间插入 $K=16$ 个可学习 **Latent Queries** $Q \in \mathbb{R}^{K\times d}$，在推理期**完全不生成未来帧**。
2. **训练期双分支序列打包（Dual-Branch MoT）**：基于 Being-H0.5 的 Mixture-of-Transformers（`InternVL3.5` 理解专家 + `Qwen3` 动作专家，纯 RGB 输入，`V-JEPA2.1` + Perceiver Resampler 提取上下文与未来观测表征）：
   - **Prior Branch（训练+部署）**：仅看当前指令、$H=4$ 历史观测与本体状态，通过 Latent Queries 推理未来控制相关因子并驱动 Flow Matching 速度场；
   - **Posterior Branch（仅训练期）**：在相同位置用未来观测 $o_{0:T}$ 经冻结 ViT + Perceiver 提取的 $z^{\text{post}}$ 替换 $Q$，揭示对动作真正有用的未来演化；
   - 两分支共享前序上下文、在单一序列中通过 **Dual-Branch Attention Mask** 互相隔离、共享位置 ID，在最后 $L=9$ 层对齐隐状态 $\mathcal{L}_{\text{align}}$。
3. **防隐空间坍塌正则**：联合优化双分支 Flow-Matching 损失与对齐损失（$w_{\text{align}}=10^{-3}$），并加入下界范数正则 $R_{\text{norm}}(h)=[\text{ReLU}(\tau-\|h\|_2)]^2$ 与负谱熵多样性正则 $R_{\text{rank}}(H)=\sum p_i \log p_i$（$w_{\text{norm}}=w_{\text{rank}}=10^{-4}$）。下游微调时仅保留 Flow-Matching 与对齐损失。
4. **仿真与真机五维盲测及 UAC 部署**：
   - 六项仿真基准：LIBERO `99.2%`、LIBERO-plus 零样本 `82.1%` / 微调 `84.8%`、RoboCasa-50 `62.1%`、GR1 `49.2%`、CALVIN `4.67 / 4.48`、RoboTwin 2.0 `90.2 / 89.6`；
   - 三平台（PND Adam-U 31 DoF、Unitree G1 26 DoF + 50 Hz AMO 全身后端、Franka FR3 13 DoF，统一配 Linkerbot O6 6-DoF 灵巧手）× 12 项高难度真机盲测（各 20 次）全面领先 $\pi_{0.5}$、Being-H0.5、Fast-WAM；配合客户端 **UAC（Universal Async Chunking，锁前缀、按延迟拼后缀）** 达成 **3–4 ms/step** 在线步长控制延迟。

## 与知识库关联

- 方法节点：[being-h07.md](../../wiki/methods/being-h07.md)
- 独立论文实体：[paper-rcl-2605-00078-being-h0-7-a-latent-world-action-model-from-egoc.md](../../wiki/entities/paper-rcl-2605-00078-being-h0-7-a-latent-world-action-model-from-egoc.md)
- 前代基座：[cn-os-being-h.md](../../wiki/entities/cn-os-being-h.md)
