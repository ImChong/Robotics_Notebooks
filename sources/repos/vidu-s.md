# Vidu-S（官方 GitHub）

> 来源归档

- **标题：** Vidu S — Real-Time Interactive, Editable, and Spatial Video Generation
- **类型：** repo
- **组织：** shengshu-ai（生数科技）
- **链接：** <https://github.com/shengshu-ai/Vidu-S>
- **项目页 / Demo：** <https://vidu.com/vidu-stream>
- **论文（S2）：** [arXiv:2609.11638](https://arxiv.org/abs/2609.11638) · [HF Papers](https://huggingface.co/papers/2609.11638)
- **论文（S1）：** [arXiv:2607.03118](https://arxiv.org/abs/2607.03118)
- **入库日期：** 2026-09-26
- **一句话说明：** 官方 **文档与概览仓**：S1/S2 README、Feishu 用户指南链、API quick-start 索引、引用 BibTeX；**不含** 模型权重或本地训练/推理代码。
- **沉淀到 wiki：** [`wiki/entities/paper-vidu-s2.md`](../../wiki/entities/paper-vidu-s2.md)

---

## 开源状态（步骤 2.5）

| 项 | 核查结论（2026-09-26） |
|----|------------------------|
| **GitHub [shengshu-ai/Vidu-S](https://github.com/shengshu-ai/Vidu-S)** | 公开；主要为 README + `figures/` |
| **可本地跑模型** | **无** — 无 checkpoint、无 `train`/`infer` 脚本 |
| **集成路径** | [Vidu Stream API](https://platform.vidu.com/vidu-stream/doc)；Agent 可选 [vidu-s-api Skill](https://github.com/shengshu-ai/vidu-s-api/tree/main/skills/vidu-s-api) |
| **产品** | [vidu.com/vidu-stream](https://vidu.com/vidu-stream) Demo ✅ |
| **结论** | **部分开源** — 文档/引用/图资产仓 + **商业 API**；**非** 可自托管训练栈 |

---

## README 技术要点（S2，归档）

1. **720p Avatar** — 25–42 FPS；流中可更新参考图；复杂指令（跳舞等）。
2. **Self-Replay Forcing** — 训练中对自生成轨迹 re-noise replay，减流式分段误差累积。
3. **S2-Editing** — 参考图驱动风格/人物/背景/试衣，保源运动。
4. **Spatial video** — 同步立体视图（沉浸显示 / VR）。
5. **TurboDiffusion + TurboServe** — 高效 attention、低比特 GEMM、内核优化、多卡 pipeline → 低成本 GPU 实时推理（**部署栈未随仓发布**）。

---

## 交叉链接

- 论文摘录：[`sources/papers/vidu_s2_arxiv_2609_11638.md`](../papers/vidu_s2_arxiv_2609_11638.md)
- 产品页：[`sources/sites/vidu_s2_stream.md`](../sites/vidu_s2_stream.md)
