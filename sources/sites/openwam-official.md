# OpenWAM 项目页（openwam-official.github.io）

- **类型**：项目静态站点
- **收录日期**：2026-09-09
- **站点**：<https://openwam-official.github.io/>
- **论文**：<https://arxiv.org/abs/2609.07398>
- **代码：** <https://github.com/OpenWAM-Official/OpenWAM>
- **权重 / 数据：** <https://huggingface.co/OpenWAM>

## 一句话

把 **World–Action Model 预训练** 拆成 **Infra（模块化栈）→ Study（六项对照问题）→ OpenWAM-α（规模化预训练模型）** 三层，并公开基础设施、评测协议、46 个检查点与数据配方。

## 开源核查（2026-09-09）

| 项 | 结论 |
|----|------|
| **代码** | **已开源** — GitHub `OpenWAM-Official/OpenWAM`（训练、部署、多基准评测客户端） |
| **权重** | **已发布** — Hugging Face `OpenWAM`（含 OpenWAM-α 与 Study 检查点） |
| **数据配方** | 项目页披露混合比例与帧数；具体数据下载经仓库 `download_benchmark_data.py` 等脚本 |

## 站点摘录要点

- **OpenWAM-Infra**：dataloaders、encoders、backbones、architectures、attention masks 可组合；统一 trainer / policy server / eval。
- **OpenWAM-Study**：Q1 架构 → Q2 骨干 → Q3 表征 → Q4 交互 → Q5 数据配方 → Q6 去噪策略；累积默认见项目页 recipe 表。
- **OpenWAM-α**：DualSystem + Wan2.2-TI2V-5B + ActionDiT + mutual mask；518.5M 帧；80-D 统一动作空间。
- **机构徽标（页内）**：NUS、Tsinghua、PKU、HKU、ZJU、CUHK、SJTU。

## 对 wiki 的映射

- 主沉淀：[OpenWAM](../../wiki/entities/paper-openwam.md)
- 原始论文档：[openwam_arxiv_2609_07398.md](../papers/openwam_arxiv_2609_07398.md)
- 代码入口：[openwam.md](../repos/openwam.md)
