# ChunkTrust（arXiv:2609.39754）

> 来源归档（paper）

- **标题：** ChunkTrust: Adapting Execution Horizons for Robot Policies with Action-Expert Evidence
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2609.39754>
- **PDF：** <https://arxiv.org/pdf/2609.39754>
- **项目页：** <https://hf618.github.io/ChunkTrust.github.io/>
- **代码：** <https://github.com/hf618/ChunkTrust>（MIT）
- **模型 / QHA 权重：** <https://huggingface.co/Niugan/ChunkTrust>
- **机构：** 清华大学；北京智源人工智能研究院（BAAI）；中国人民大学；深圳技术大学；合肥工业大学；江南大学；重庆大学；香港中文大学
- **入库日期：** 2026-10-01
- **一句话说明：** 把 execution horizon 当作由 action-expert 证据推断的隐变量；**训练-free AHS** 融合 chunk 内去噪速度谱稳定性与 executed history 边界连续性，并用 episode-local Beta 记忆跟踪阶段偏好；可选 **QHA** 从冻结策略特征学习 horizon 先验，与在线证据融合，**不改 base policy 权重**。

## 开源状态

- **已开源：** [hf618/ChunkTrust](https://github.com/hf618/ChunkTrust) 含 `AHS` / `HybridQHARuntimeSelector`、RoboTwin 2.0 / RoboCasa / 真机集成文档、`examples/quickstart.py` 与 CI。
- **权重：** Hugging Face [Niugan/ChunkTrust](https://huggingface.co/Niugan/ChunkTrust) 发布 QHA head checkpoint；基础 VLA 仍走各 benchmark 原训练栈。
- **数据：** 仿真训练数据来自 RoboTwin 2.0 与 RoboCasa GR1 Tabletop 官方管线（见仓库 `docs/backends.md`）。

## Abstract（arXiv）

Robot foundation policies predict action chunks, but how many actions to execute before replanning depends on the current task phase. ChunkTrust treats execution horizon as a latent variable from action-expert evidence. AHS combines intra-chunk spectral stability of generation traces with inter-chunk continuity; an online Beta posterior with kernel forgetting tracks preferences. QHA optionally learns a context-conditioned dense prior fused at deployment while the base policy stays frozen. Gains on RoboTwin2.0, RoboCasa GR1 Tabletop, and four real household tasks; ablations on evidence, memory, and learned prior.

## 核心摘录

1. **动机：** 固定 execution horizon 在 reach / contact / handover 等阶段次优；1600 条固定 \(K{=}H{=}50\) 的 π0 RoboTwin rollout 显示失败 episode 的 **谱不稳定** 与 **边界速度波动** 中位数更高，双高风险时失败率约 **97.4%**。
2. **Intra-chunk 证据：** 记录 flow/denoising 各步 velocity prefix，沿 action horizon 做 FFT，比较高频能量比例在采样步间的 RMS 变化 → \(z_{\mathrm{intra}}(k)\)。
3. **Inter-chunk 证据：** 将候选前缀 \(k\) 与 executed history 拼接，度量边界附近动作速度变化 → \(u_{\mathrm{inter}}(k)\)；与 BID / DVAC / GeoAAC 等「单信号」方法对照，强调 **生成动力学 + 已执行运动兼容** 配对。
4. **AHS：** 归一化证据 → 每个候选 horizon 的 Beta 可靠性 → 选择分布；episode-local 记忆 + kernel forgetting 适应阶段切换；**零训练**、evaluation-time wrapper。
5. **QHA：** 冻结 backbone 的 context / action-expert 特征 → 轻量 query head 学习 dense horizon 先验；部署时与当前 AHS 证据 + Beta 记忆 **一次融合、一次记忆更新**（避免双重 posterior 计数）。
6. **仿真：** π0.5 全 50 RoboTwin2.0 任务 **56.70% → 63.50%**（AHS）；8 任务 π0.5 **29.63% → 39.06%**（AHS+QHA）；Qwen3GR00T RoboCasa 24 任务 **47.83% → 57.50%**（AHS）。
7. **真机：** AgileX COBOT Magic，π0.5，四家务任务 equal-task 平均 normalized process score **50.4% → 57.5%**（AHS）。

**对 wiki 的映射**

- [paper-chunktrust](../../wiki/entities/paper-chunktrust.md)
- [chunktrust-project.md](../sites/chunktrust-project.md)
- [chunktrust.md](../repos/chunktrust.md)
- [receding-horizon-policy-execution](../../wiki/concepts/receding-horizon-policy-execution.md)
- [action-chunking](../../wiki/methods/action-chunking.md)
