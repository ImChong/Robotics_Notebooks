# Dita 项目页（robodita.github.io）

> 来源归档（site）

- **标题：** Dita: Scaling Diffusion Transformer for Generalist Vision-Language-Action Policy
- **类型：** site（论文项目页 + 真机/SimplerEnv/LIBERO 视频）
- **URL：** <https://robodita.github.io/>
- **论文：** [arXiv:2503.19757](https://arxiv.org/abs/2503.19757)（canonical）；前序 [arXiv:2410.15959](https://arxiv.org/abs/2410.15959)
- **代码：** <https://github.com/RoboDita/Dita>
- **机构：** 上海人工智能实验室；浙江大学；商汤科技；香港中文大学 MMLab；北京大学；清华大学；中科院 HKISI 等
- **入库日期：** 2026-09-28
- **一句话说明：** 展示 **Dita** 真机 10-shot 长时域任务、背景/光照方差鲁棒性，以及 SimplerEnv / LIBERO 评测视频。

## 开源核查（步骤 2.5，2026-09-28）

| 资源 | 状态 |
|------|------|
| GitHub | **已开源** — [RoboDita/Dita](https://github.com/RoboDita/Dita)（页内 Paper + Code 按钮一致） |
| 权重 | README / 项目页链 Google Drive checkpoint |
| 2410 旧站 | [zhihou7.github.io/dit_policy_vla](https://zhihou7.github.io/dit_policy_vla/) — 早期 Diffusion Transformer Policy 页，现以 RoboDita 为主 |

**判定：已开源可微调/评测**（全量 OXE 预训练依赖 S3 与多机）。

## 公开要点（编译自项目页，2026-09-28）

- **真机：** 开抽屉、叠碗、倒豆/倒水、开盒取物等多步任务；**Variance** 小节含背景、桌布、光照变化。
- **SimplerEnv：** OXE 预训练后在 Real-to-Sim 抽屉/可乐罐/蔬果等子任务视频。
- **LIBERO：** Long / Spatial / Object / Goal 四套件 finetune 示例。

## 关联资料

- 论文归档：[`sources/papers/dita_arxiv_2410_15959.md`](../papers/dita_arxiv_2410_15959.md)
- 代码归档：[`sources/repos/robodita_dita.md`](../repos/robodita_dita.md)
- Wiki：[`wiki/entities/paper-dita-scaling-diffusion-transformer-vla.md`](../../wiki/entities/paper-dita-scaling-diffusion-transformer-vla.md)
