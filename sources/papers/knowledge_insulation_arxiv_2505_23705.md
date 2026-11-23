# Knowledge Insulating Vision-Language-Action Models（arXiv:2505.23705）

> 来源归档（ingest）

- **标题：** Knowledge Insulating Vision-Language-Action Models: Train Fast, Run Fast, Generalize Better
- **短名：** Knowledge Insulation / KI
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2505.23705>
- **项目页：** <https://www.pi.website/research/knowledge_insulation>
- **机构：** 物理智能（Physical Intelligence）
- **入库日期：** 2026-09-28
- **核对范围：** 2026-11-22 通读 `https://arxiv.org/html/2505.23705v1` 论文全文（Abstract、Sec. 1–7 及 Appendix A–C）与 `Physical-Intelligence/openpi` `README.md`；`https://www.pi.website/research/knowledge_insulation` 返回 HTTP 429 Vercel Security Checkpoint（反爬门禁），未尝试绕过；未在本地 GPU 上复现训练。
- **通读：** 已完成
- **一句话说明：** 用 FAST 离散动作训练 VLM 骨干，同时让 flow 动作专家出连续动作且梯度不回传骨干。

## 开源状态（步骤 2.5，2026-09-28 / 2026-11-22 复核）

- **部分开源**：[openpi](https://github.com/Physical-Intelligence/openpi) `README.md` 写明 π₀.₅ 使用 knowledge insulation 预训练，并发布 `pi05_base`、`pi05_libero`、`pi05_droid`（DROID 上按 KI 微调）等检查点；同一 README 注明该仓库当前对 π₀.₅ 训练与推理只支持 flow matching 头。
- 维护者在 [openpi#649](https://github.com/Physical-Intelligence/openpi/issues/649) 说明：预训练 π₀.₅ 检查点用了 KI；`train.py` 微调只对 action expert 的 flow matching 损失反传，梯度会进入 VLM 骨干，**未实现** FAST token 与 stop-gradient 的 KI 微调。

## 核心摘录（面向 wiki 编译）

- **架构与损失（Sec. 3, 5 & Appendix B）：** 基于 PaliGemma 2B VLM 骨干（`width=2048, depth=18, mlp_dim=16384, num_heads=18, num_kv_heads=1, head_dim=256`）与 300M 参数流匹配动作专家（`width=1024, mlp_dim=4096`，动作块长度 $H=50$）；联合目标 $\mathcal{L}_{\text{CO-VLA}}$ 同时包含 VLM 文本 + FAST 离散动作 token 的交叉熵损失与连续动作流的 MSE 损失（引入 stop-gradient 后可直接设 $\alpha=1$）。
- **梯度隔离与掩码（Sec. 5.2 & Appendix B）：** 在注意力层对动作专家向骨干的 Key/Value 投影插入 `sg`（$Q_a(X_a)\,\text{sg}(K_b(X_b)^T)$ 与 $P_{ab}\,\text{sg}(V_b(X_b))$），并设置注意力掩码 $A$ 使离散 FAST 动作 token 与连续动作 token 互不可见，从而在不损伤预训练骨干的前提下将推理交由小专家（约 10 Hz，vs $\pi_0$-FAST 约 1.3 Hz / 750 ms）。
- **本体状态与 VLM 共训（Appendix A.3 & C）：** 对比文本状态、特殊 token 状态与连续线性投影状态（文本状态最贴近预训练，但消耗最多 token；实验表明文本与连续状态均可配合 KI 工作，特殊 token 较差）；通才模型在 12 种机器人构型及 OXE 数据上训练，并共训 CapsFusion / COCO / Cambrian-7M / PixMo / VQAv2 与目标定位 bbox 数据。
- **定量结果与代价（Sec. 6–7 & Table 1）：** 真实任务包括 `items in drawer`（未见厨房，5 分制）、`T-shirt folding`（5 分制）、`table bussing`（12 分制）与 4 项移动双臂任务；DROID 桌面任务达 $0.55 \pm 0.09$（优于 $\pi_0$ $0.49 \pm 0.09$ 与 $\pi_0$-FAST $0.45 \pm 0.09$）；通才微调版在 LIBERO-Spatial / Object / Goal / 10 / 90 分别达 98.0% / 97.8% / 95.6% / 85.8% / 96.0%。训练同时算离散+连续头使单步算力增加约 20%，但收敛步数大幅减少（通才 bussing 训练步数约为 $\pi_0$ 的 1/7.5）。
- **对 wiki 的映射：** [paper-knowledge-insulation](../../wiki/entities/paper-knowledge-insulation.md)
