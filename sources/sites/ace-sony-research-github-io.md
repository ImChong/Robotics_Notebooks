# sonyresearch.github.io/ace_public（Sony AI Ace 项目 / 补充材料页）

- **标题：** Ace Supplementary Material — Outplaying Elite Table Tennis Players with an Autonomous Robot
- **类型：** site / project-page / supplementary
- **URL：** <https://sonyresearch.github.io/ace_public/>
- **配套论文：** [Nature s41586-026-10338-5](https://doi.org/10.1038/s41586-026-10338-5) — 归档见 [`sources/papers/sony_ace_nature_2026.md`](../papers/sony_ace_nature_2026.md)
- **代码 / 数据入口：** <https://github.com/SonyResearch/ace_public> — 归档见 [`sources/repos/ace-public.md`](../repos/ace-public.md)
- **Ace 品牌站：** <https://ace.ai.sony>
- **入库日期：** 2026-09-16

## 一句话摘要

Sony Research **Ace** 官方补充材料站：完整对局视频、GCS 凝视控制、触网反应、发球与专家点评；链向 GitHub 上的 **match 数据集** 与 **近似 RL/感知伪代码**。

## 公开信息要点（截至入库日）

- **机构：** Sony AI / Sony Research Inc.
- **页内板块：**
  - **Match Highlights** — 精选回合
  - **Gaze Control System** — 三 GCS 跟踪 spin 演示
  - **Figure 4 Net Bounce** — 触网后 49 ms 级反应
  - **Serves** — 遗传算法发球库
  - **Expert Comment** — 奥运选手 Kinjiro Nakamura 等点评
  - **Full Matches** — 对 5 精英 + 2 职业完整 BO3/BO5
  - **Additional Experiments** — 同行评议后追加实验（链向 ace.ai.sony）
- **数据页：** [`/data/`](https://sonyresearch.github.io/ace_public/data/) — `match_data.csv` 字段说明（post-event ball pos/vel/spin）
- **伪代码页：** [`/pseudo_code/`](https://sonyresearch.github.io/ace_public/pseudo_code/) — SAC 训练环、rollout worker、FAOC、发球 GA、GCS 伪实现

## 源码开放核查（步骤 2.5）

| 链接 | 结论 |
|------|------|
| Footer / README → GitHub | ✅ `SonyResearch/ace_public` |
| 数据集 | ✅ CSV 可下载 |
| 训练 / 部署代码 | ❌ 仅 **approximate pseudo code**，非生产可运行栈 |
| 权重 / 硬件 | ❌ 未列 |

→ **部分开源**：复现需自建硬件与全栈；官方提供 **数据分析 + 算法结构参考**。

## 为何值得保留

- **非 PDF 证据：** 完整对局与 GCS 跟踪比表格更直观呈现 **spin 感知 + 真机竞技** 能力边界。
- **数据格式权威：** 坐标系（桌心原点、x 朝人类侧）与 CSV schema 是下游分析的唯一官方说明。
- **与 Nature / GitHub 三角互证：** 视频、数据、伪代码入口一致。

## 关联资料

- 论文归档：[`sources/papers/sony_ace_nature_2026.md`](../papers/sony_ace_nature_2026.md)
- 代码仓库：[`sources/repos/ace-public.md`](../repos/ace-public.md)
