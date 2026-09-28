# Compression is Routing: Reconstruction Error as an Intrinsic Signal for Modular Language Models（arXiv:2512.16963）

> 来源归档（ingest）

- **标题：** Compression is Routing: Reconstruction Error as an Intrinsic Signal for Modular Language Models
- **类型：** paper / modular LLM / MoE routing / context compression / continual learning / technical report
- **arXiv：** <https://arxiv.org/abs/2512.16963>（PDF：<https://arxiv.org/pdf/2512.16963.pdf>）
- **ScienceStack：** <https://www.sciencestack.ai/paper/2512.16963>
- **CatalyzeX：** <https://www.catalyzex.com/paper/compression-is-routing-reconstruction-error>
- **ResearchGate：** <https://www.researchgate.net/publication/398936728_Compression_is_Routing_Reconstruction_Error_as_an_Intrinsic_Signal_for_Modular_Language_Models>
- **作者：** Zhongpan Tang（独立研究者；联系 tangzhongp@gmail.com）
- **机构：** 独立研究（论文未列机构 affiliation）
- **入库日期：** 2026-09-28
- **一句话说明：** 提出 **「Compression is Routing」**：87M 端到端 Transformer 自编码器将 **512 token → 8 latent（64× 序列压缩）**；在 **code 域** TRA **99.47%**，Wiki **47.76%**，随机 **0.57%**——重建误差可作为 **无显式门控** 的域指纹，调度模块化专家并缓解灾难性遗忘与 KV VRAM 压力。

## 开源状态（arXiv + 聚合页核查，2026-09-28）

- **确认未开源：** arXiv HTML/PDF **未列 GitHub / 项目页 / 权重**；ScienceStack、CatalyzeX、ResearchGate 仅为论文索引/外链，**无可运行实现**。作者自述为 **Independent Researcher**，受算力限制未做更大规模消融；报告意图是邀请社区复现与扩展。

## 摘要级要点

- **动机：** 单体 LLM 面临上下文长度、推理成本、持续学习遗忘；传统 MoE 依赖 **显式训练的门控网络**，混合域输入时可解释性弱。
- **架构：** 非对称 Transformer AE：Encoder 512→8 latent；Decoder 仅读 \(z\) 与 **与 \(x\) 内容隔离** 的辅助 \(m\)，强制信息瓶颈；GPT-2 tokenizer；**codeparrot** 预训练。
- **核心指标 TRA：** token 级重建准确率（严格位置匹配，非模糊语义相似）。
- **几何：** t-SNE/UMAP 显示 code vs Wiki latent **几乎正交可分**；PCA 显示 code 内在维 \(\approx 200 < 512\)。
- **应用叙事：** （1）**增量挂载** 域专家、冻结旧 compressor；（2）embedding 后 **64× KV 压缩** 缓解超长上下文 VRAM。
- **局限：** 固定块长 \(L=512\) 在域切换边界有 **滞后**；交错分布（博客内嵌代码）落入 **灰色 TRA 带**。

## 核心论文摘录（MVP）

### 1) 端到端语言自编码器

- **链接：** §2.1–2.2；Eq. (1)
- **摘录要点：** \(x \to \text{Encoder} \to z \to [z,m] \to \text{Decoder} \to \hat{x}\)；CE loss；\(m\) 与 \(x\) 等长但 **无 \(x\) 内容**，Decoder **不可见** 原始 \(x\)。
- **对 wiki 的映射：**
  - [Compression is Routing](../../wiki/entities/paper-compression-is-routing.md) — 架构与瓶颈设计。
  - [MemoryWAM](../../wiki/entities/paper-memorywam.md) — 同属 **长上下文压缩** 叙事（gist/latent vs 本文 AE 路由）。

### 2) 三级 TRA 阶梯（路由信号）

- **链接：** §3.3；Tab. 1
- **摘录要点：** ID code **99.47%** → Semi-OOD Wiki **47.76%**（共享词表统计，触发 **增量专家**）→ Full-OOD random **0.57%**（强拒噪）。
- **对 wiki 的映射：**
  - [Compression is Routing](../../wiki/entities/paper-compression-is-routing.md) — 实验主表。
  - [AtomicVLA](../../wiki/entities/paper-atomicvla.md) — 机器人侧 **显式 routing encoder + SG-MoE** 对照。

### 3) 模块化与 VRAM 讨论

- **链接：** §4.1–4.2
- **摘录要点：** 相对单体全量微调+Replay，新域可 **只训练新 residual compressor** 并冻结旧模块；latent 序列长 8 替代 512 token 的 KV 占用。
- **对 wiki 的映射：**
  - [Compression is Routing](../../wiki/entities/paper-compression-is-routing.md) — 工程含义。
  - [GSR / ParaVLA](../../wiki/entities/paper-gsr-paravla.md) — VLA **路由脆弱性** 的另一解（语义重绑 vs 重建误差路由）。

### 4) 局限与未来

- **链接：** §5
- **摘录要点：** 块内域突变 **hysteresis**；\(M=8\) 为当前数据上 **近无损最小** latent 长度（\(M=2,4\) 明显掉点）；更高压缩比（128×）待社区验证。
- **对 wiki 的映射：**
  - [Compression is Routing](../../wiki/entities/paper-compression-is-routing.md) — 结论与机器人读者读法。

## BibTeX

```bibtex
@misc{tang2025compressionisrouting,
  title         = {Compression is Routing: Reconstruction Error as an Intrinsic Signal for Modular Language Models},
  author        = {Tang, Zhongpan},
  year          = {2025},
  eprint        = {2512.16963},
  archivePrefix = {arXiv},
  primaryClass  = {cs.CL},
  url           = {https://arxiv.org/abs/2512.16963}
}
```

## 对 wiki 的映射

- 主实体页：[wiki/entities/paper-compression-is-routing.md](../../wiki/entities/paper-compression-is-routing.md)
