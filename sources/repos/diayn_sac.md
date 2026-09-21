# DIAYN（SAC 实现）— 官方技能发现代码

> 来源归档（ingest · arXiv:1802.06070 配套）

- **标题：** DIAYN implementation in ben-eysenbach/sac
- **类型：** repo
- **官方入口：** <https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md>
- **项目页：** <https://sites.google.com/view/diayn/>
- **关联论文：** [Diversity is All You Need](../papers/bfm_awesome_diayn_iclr_2018.md)（arXiv:1802.06070，ICLR 2018）
- **入库日期：** 2026-09-21
- **一句话说明：** DIAYN 基于 **Soft Actor-Critic（SAC）** 的无监督技能发现参考实现；伪奖励 $r_z(s)=\log q_\phi(z|s)-\log p(z)$ 与 discriminator 更新逻辑见 `DIAYN.md`。

## 第三方复现（非官方）

- <https://github.com/alirezakazemipour/DIAYN-PyTorch> — PyTorch 社区复现，便于现代 RL 栈对照

## 开源边界（步骤 2.5）

| 状态 | 说明 |
|------|------|
| **已开源** | 官方 README + 项目页视频/代码链接可访问 |
| **环境** | 经典 MuJoCo / Gym 任务；与现代 Isaac 人形栈 **非直接可插** |

## 对 wiki 的映射

- [paper-bfm-30-diayn.md](../../wiki/entities/paper-bfm-30-diayn.md)
- [behavior-foundation-model.md](../../wiki/concepts/behavior-foundation-model.md)

## 参考来源（原始）

- 官方 DIAYN 文档：<https://github.com/ben-eysenbach/sac/blob/master/DIAYN.md>
- 项目页：<https://sites.google.com/view/diayn/>
- 论文：<https://arxiv.org/abs/1802.06070>
