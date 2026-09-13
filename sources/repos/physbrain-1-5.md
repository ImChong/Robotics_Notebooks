# PhysBrain-1.5

> 来源归档

- **标题：** PhysBrain 1.5 — From General VLMs to Physical Foundation Model
- **类型：** repo（官方文档与技术报告入口）
- **机构：** 机智赛博（DeepCybo）；北京中关村学院；中关村人工智能研究院（ZGCI / ZGCA）
- **链接：** <https://github.com/DeepCybo-PhysAI/PhysBrain-1.5>
- **项目页：** <https://deepcybo-physai.github.io/PhysBrain-1.5/>
- **技术报告：** <https://github.com/DeepCybo-PhysAI/PhysBrain-1.5/blob/main/tech_report.pdf>
- **论文（前作）：** <https://arxiv.org/abs/2512.16793>
- **Hugging Face：** <https://huggingface.co/collections/DeepCybo/physbrain-15>
- **Demo：** <https://huggingface.co/spaces/hugging-apps/physbrain1-5-8b-demo>
- **评测工具：** <https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit>
- **入库日期：** 2026-09-13
- **许可证：** 仓内未声明 LICENSE 文件（截至核查日）
- **代码 / 开源状态：** **部分开源** — **已发布** 技术报告 PDF + README + 项目页资产；**HF 2B/8B 权重**与 Demo；**未见** 预训练 / SFT 训练脚本与数据管线
- **一句话说明：** PhysBrain 1.5 官方 GitHub 落地页：读技术报告与下载权重的入口；可复现推理走 Hugging Face，28 benchmark 评测走 PhysBrainEvalKit。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md`](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)
- **交叉归档：** [physbrain-1-5-github-io.md](../sites/physbrain-1-5-github-io.md)、[physbrain_1_5_technical_report_2026.md](../papers/physbrain_1_5_technical_report_2026.md)、[physbrain-eval-kit.md](./physbrain-eval-kit.md)

---

## 仓内结构（2026-09-13 快照）

| 路径 | 作用 |
|------|------|
| `README.md` | 能力摘要、榜单、HF 权重链接、引用 |
| `tech_report.pdf` | 技术报告全文 |
| `gh-pages/` | 项目站静态资源（架构图、榜单图等） |

## Hugging Face 权重

| Checkpoint | 链接 |
|------------|------|
| PhysBrain 1.5-8B | [DeepCybo/PhysBrain1.5-8B](https://huggingface.co/DeepCybo/PhysBrain1.5-8B) |
| PhysBrain 1.5-2B | [DeepCybo/PhysBrain1.5-2B](https://huggingface.co/DeepCybo/PhysBrain1.5-2B) |

---

## 对 wiki 的映射

- 实体页：[PhysBrain](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)
- 方法交叉：[VLA](../../wiki/methods/vla.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md)
- 同机构对照：[POT-VLA](../../wiki/entities/paper-pot-vla.md)、[Human-as-Humanoid](../../wiki/entities/paper-human-as-humanoid.md)
- 同族对照：[ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md)、[RynnBrain 1.1](../../wiki/entities/paper-rynnbrain-1-1.md)
