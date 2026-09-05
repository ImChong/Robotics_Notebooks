# DexHoldem（arXiv:2605.18727）

> 来源归档（ingest）

- **标题：** DexHoldem: Playing Texas Hold'em with Dexterous Embodied System
- **类型：** paper / dexterous-manipulation / embodied-agent / benchmark / teleoperation / vla
- **arXiv abs：** <https://arxiv.org/abs/2605.18727>
- **PDF：** <https://arxiv.org/pdf/2605.18727>
- **HTML：** <https://arxiv.org/html/2605.18727v1>
- **项目页：** <https://dexholdem.github.io/Dexholdem/> — 归档见 [`sources/sites/dexholdem-github-io.md`](../sites/dexholdem-github-io.md)
- **代码：** <https://github.com/DexHoldem/Dexholdem-Policy> — 归档见 [`sources/repos/dexholdem-policy.md`](../repos/dexholdem-policy.md)（**已开源**，截至 2026-09-05 未附 LICENSE）
- **技能仓：** <https://github.com/DexHoldem/DexHoldemSKills> — [`sources/repos/dexholdem-skills.md`](../repos/dexholdem-skills.md)
- **数据集：** <https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot> — [`sources/datasets/texaspokerrobot.md`](../datasets/texaspokerrobot.md)（CC BY 4.0）
- **机构：** 香港大学（HKU）；加州大学伯克利分校（UC Berkeley，Yuexiang Zhai / Yi Ma 访问）
- **作者：** Feng Chen\*†、Tianzhe Chu\*、Li Sun\*、Pei Zhou\*、Zhuxiu Xu、Shenghua Gao、Yuexiang Zhai、Yanchao Yang、Yi Ma（\* 共同一作；† 项目负责人）
- **发表 / 上传：** arXiv 2026-05（2605.18727）；项目页 BibTeX 标 2026
- **入库日期：** 2026-09-05
- **一句话说明：** 用德州扑克桌面把「指令条件灵巧执行」和「结构化牌桌状态恢复」绑在同一套 ShadowHand + UR10e 真机协议上。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [dexholdem.github.io/Dexholdem](https://dexholdem.github.io/Dexholdem/) | 视频、三张榜、Resources |
| 官方策略仓 | [DexHoldem/Dexholdem-Policy](https://github.com/DexHoldem/Dexholdem-Policy) | 六条可复现训练配方 + ZeroMQ 部署 |
| Agent skill | [DexHoldem/DexHoldemSKills](https://github.com/DexHoldem/DexHoldemSKills) | `npx skills add` 后走 perceive–route–execute |
| 数据集 | [Winniechen2002/TexasPokerRobot](https://huggingface.co/datasets/Winniechen2002/TexasPokerRobot) | 1,470 条 `.npz`，约 378 GB |

## 开源状态（步骤 2.5，截至 2026-09-05 复核）

- **已开源：** 项目页 Resources 同时链到 GitHub org、Policy 仓、Skills 仓与 Hugging Face 数据集。Policy 仓含 `scripts/train_*.sh`、`workflow/download_data.py`、`deploy_policy.py`、`robot_client.py`。
- **边界：** 两仓无 LICENSE 文件（GitHub `license: null`）。公开配方覆盖 DP / ACT / RDT 六条；π 系走 OpenPI 桥，不在六条 public recipe 里。Being-H 已微调但因振荡未进主榜。
- **数据：** TexasPokerRobot 标 CC BY 4.0；HF Viewer 只预览 manifest，原始 episode 需 `allow_pickle=True`。
- **处理：** wiki 写「已开源可运行」；源码运行时序图对齐 `workflow/` + `deploy_policy.py` + Skills sandbox。

## 摘要级要点

- **问题：** 具身 agent 榜多在仿真/夹爪；灵巧榜多测孤立技能。缺一块同时要语义接地、顺序状态、薄牌/筹码接触的真机协议。
- **主张：** 德州扑克桌面不是测扑克智能，而是把指令条件灵巧执行、场面保持、结构化状态恢复绑在同一物理场景。
- **方法：** 14 个原语 × 105 条遥操作（100/5 划分）；共享 3 路 RealSense RGB-D + 30 维关节接口；四级结果量表（SP / DC / TF / DF）；36 题感知 bench；3 条闭环案例。
- **结果要点：**
  - π₀.₅ TCR **61.2%**，π₀.₅ 与 π₀ SPSR 并列 **47.5%**
  - 芯片类原语远弱于 pickup（π 系 pickup 100%，chip pull SPSR 仅 15%）
  - 感知严格 exact match 最高 **34.3%**（Opus 4.7）；field-wise 最高 **66.8%**（GPT 5.5）；CB / OCI 是路由关键短板
  - 夹爪预训练 RDT 在 10% 数据上验证损失只降 1.2%，不像语言/视觉那样少样本迁移
- **局限：** 固定桌面与 ShadowHand–UR；1,470 条相对预训练规模很小；仿真重建不能替代真机接触；系统级只有 3 条案例、不报成功率。

## 核心摘录（面向 wiki 编译）

### 1) 完成任务 ≠ 场面还能继续

四级量表把「牌拿起来了但旁边筹码被扫飞」记成 disruptive completion。π₀.₅ SPSR 47.5% → TCR 61.2%；RDT 30.0% → 46.2%。长程闭环吃的是 SPSR，不是 TCR。

### 2) 感知子能力加不起来

GPT 5.5 field-wise 66.8%，严格 Overall 只有 31.5%。盲注/轮次标记接近饱和，当前下注筹码与对手筹码字典峰值 45.8% / 43.8%。错一个字段就路由错。

### 3) 闭环被等待和恢复主导

三条 GPT 5.5 + π₀ 案例：轨迹 (ii) 54 个状态里 26 次 wait、22 次策略原语、1 次恢复。系统级贡献是暴露误差累积，不是统计显著胜率。

### 4) 复现入口

`workflow/download_data.py` → `organize_data.py`（每原语 5 条 val）→ `precompute_features.py`（DinoV2 / SigLIP）→ `scripts/train_*.sh`。部署：`deploy_policy.py --port 13579` + `robot_client.py`。Agent：`npx skills add DexHoldem/DexHoldemSKills`。

## 对 wiki 的映射

- 实体页：[wiki/entities/paper-dexholdem.md](../../wiki/entities/paper-dexholdem.md)
- 交叉：[操作](../../wiki/tasks/manipulation.md)、[接触丰富操作](../../wiki/concepts/contact-rich-manipulation.md)、[DexBench](../../wiki/entities/dexbench.md)、[具身评测枢纽](../../wiki/overview/hub-embodied-eval-benchmark.md)、[评测选型闭环](../../wiki/queries/embodied-eval-benchmark-selection-loop.md)、[π₀](../../wiki/methods/π0-policy.md)、[VLA](../../wiki/methods/vla.md)、[Diffusion Policy](../../wiki/methods/diffusion-policy.md)
