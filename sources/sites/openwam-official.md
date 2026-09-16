# OpenWAM 项目页（openwam-official.github.io）

- **类型**：项目静态站点
- **收录日期**：2026-09-09
- **站点**：<https://openwam-official.github.io/>
- **论文**：<https://arxiv.org/abs/2609.07398>
- **代码：** <https://github.com/OpenWAM-Official/OpenWAM>
- **GitHub 组织：** <https://github.com/OpenWAM-Official>
- **权重 / 数据：** <https://huggingface.co/OpenWAM>
- **HF 论文页：** <https://huggingface.co/papers/2609.07398>
- **OpenWAM-α Foundation：** <https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model>

## 一句话

把 **World–Action Model 预训练** 拆成 **Infra（模块化栈）→ Study（六项对照问题）→ OpenWAM-α（规模化预训练模型）** 三层，并公开基础设施、评测协议、46 个检查点与数据配方。

## 开源核查（2026-09-16）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — GitHub 组织 [`OpenWAM-Official`](https://github.com/OpenWAM-Official) / 主仓 `OpenWAM`（训练、部署、多基准评测客户端） |
| **权重** | **已发布** — HF [`OpenWAM`](https://huggingface.co/OpenWAM)（**46** 检查点）；Foundation 见 [`OpenWAM-Alpha-Pretrain-Foundation-Model`](https://huggingface.co/OpenWAM/OpenWAM-Alpha-Pretrain-Foundation-Model) |
| **HF 论文页** | [`papers/2609.07398`](https://huggingface.co/papers/2609.07398) |
| **数据配方** | 518.5M 帧：机器人 70%（真机 40% + 仿真 30%）+ egocentric 30%；下载经 `download_benchmark_data.py` 等 |

## 站点摘录要点

- **OpenWAM-Infra**：dataloaders、encoders、backbones、architectures、attention masks 可组合；统一 trainer / policy server / eval。
- **OpenWAM-Study**：Q1 架构 → Q2 骨干 → Q3 表征 → Q4 交互 → Q5 数据配方 → Q6 去噪策略；累积默认见项目页 recipe 表。
- **OpenWAM-α**：DualSystem + Wan2.2-TI2V-5B + ActionDiT + mutual mask；518.5M 帧；80-D 统一动作空间。
- **机构徽标（页内）**：NUS、Tsinghua、PKU、HKU、ZJU、CUHK、SJTU。

## 对 wiki 的映射

- 主沉淀：[OpenWAM](../../wiki/entities/paper-openwam.md)
- 原始论文档：[openwam_arxiv_2609_07398.md](../papers/openwam_arxiv_2609_07398.md)
- 代码入口：[openwam.md](../repos/openwam.md)
