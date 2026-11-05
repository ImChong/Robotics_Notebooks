# VQ-BeT: Behavior Generation with Latent Actions（arXiv:2403.03181）

> 来源归档（ingest）

- **标题：** Behavior Generation with Latent Actions
- **短名：** VQ-BeT（Vector-Quantized Behavior Transformer）
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2403.03181>（v1 提交 2024-03-05；PDF 为 v2，2024-06-28）
- **PDF：** <https://arxiv.org/pdf/2403.03181>
- **项目页：** <https://sjlee.cc/vq-bet/>
- **作者：** Seungjae Lee, Yibin Wang, Haritheja Etukuru, H. Jin Kim, Nur Muhammad Mahi Shafiullah, Lerrel Pinto（arXiv 作者列表；PDF 标注后两位为共同通讯作者）
- **发表：** PDF 页脚为 Proceedings of the 41st International Conference on Machine Learning（ICML 2024）
- **官方代码：** <https://github.com/jayLEE0301/vq_bet_official>（项目页链接；本次**未**克隆阅读其源码）
- **LeRobot 实现：** `src/lerobot/policies/vqbet/`（固定提交 9e5ee9859480e425a97a6836bb3fee479539beb5）；LeRobot 文档 `docs/source/policy_vqbet_README.md` 仅含项目页链接与 BibTeX
- **入库日期：** 2026-11-05
- **一句话说明：** 用 Residual VQ-VAE 把连续动作（chunk）离散成两层码字，再用 GPT 式 Transformer 预测码字与连续 offset，单次前向生成多峰动作。

## 核心摘录（面向 wiki 编译，均读自 arXiv PDF）

- **动机：** BeT 用 k-means 离散化动作，在高维动作空间或动作 chunk 上扩展性差、且无梯度信息；VQ-BeT 改用层级向量量化（Residual VQ）做动作 tokenizer。
- **§3.2 动作（chunk）离散化：** 采用 Residual VQ-VAE（Zeghidour et al., 2021）；实验中 Nq=2 层残差量化即够用。输入向量先过第一层量化得 z_q1，残差再逐层送入下一层。
- **§3.3 / Eq.4–7：** 先训练 Residual VQ，再训练 GPT 式 Transformer 预测各层码字（Lcode）与连续 offset（Loffset，用于补回量化损失的精度），总损失 L = Lcode + Loffset。
- **§3.4：** 同一架构覆盖条件（目标条件，Oh × Og → An）与非条件（Oh → An）两种设定。
- **贡献声明（§1）：** VQ-BeT 是单次前向模型，相对多步扩散模型在仿真中快 5×、真机上快 25×；真机长时程任务上宣称较此前工作提升 73%（作者表述，未核对原表）。
- **真机实验注意（§5）：** 作者在其低成本移动操作机器人上发现 Diffusion Policy 的 receding-horizon 执行完全失效（控制噪声使开环 3 步就出分布），因此所有策略都按完全闭环 rollout，由此得到 25× 推理时间差（附录 Table 11 对比了闭环与 receding horizon）。
- **附录 B.1：** 把 Residual VQ 码本组合数扩大 10 至 250 倍，对性能影响很小（Table 12 的结论）。

## 对 wiki 的映射

- [paper-vq-bet-behavior-generation-latent-actions](../../wiki/entities/paper-vq-bet-behavior-generation-latent-actions.md)
- 相关方法页：[action-chunking](../../wiki/methods/action-chunking.md)、[diffusion-policy](../../wiki/methods/diffusion-policy.md)

## 当前提炼状态

- [x] arXiv 摘要页与 PDF 正文（§1、§3.2–3.4、§5、附录 B.1）已读
- [x] LeRobot `policies/vqbet/` 源码已读
- [ ] 官方仓库 `vq_bet_official` 源码未读
