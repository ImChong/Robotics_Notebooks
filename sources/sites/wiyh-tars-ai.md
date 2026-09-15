# World In Your Hands（项目页）

> 来源归档（site）

- **标题：** World In Your Hands — Large-Scale and Open-source Ecosystem for Learning Human-centric Manipulation in the Wild
- **类型：** site / dataset / benchmark / hardware
- **机构：** 它石智航（TARS Robotics）
- **链接：** <https://wiyh.tars-ai.com/>
- **论文：** <https://arxiv.org/abs/2512.24310>
- **代码 / Devkit：** <https://github.com/tars-robotics/World-In-Your-Hands>
- **数据集：** <https://huggingface.co/datasets/tars-robotics/WIYH>
- **入库日期：** 2026-09-15
- **一句话说明：** WIYH 官方门户：Oracle Suite 可穿戴采集硬件说明、全量 ~1000 h 多模态人类操作数据下载、教程与 benchmark 入口。

## 开源状态（步骤 2.5）

核查日：**2026-09-15**（项目页 / GitHub README / Hugging Face）。

| 产物 | 状态 |
|------|------|
| 项目页 `wiyh.tars-ai.com` | **已开放** — 数据集说明、下载、教程 |
| GitHub `tars-robotics/World-In-Your-Hands` | **已开源** — devkit（`wiyh.py`）、`wiyh2lerobot` 转换、`wiyh_tutorial.ipynb`；CC BY-NC-SA 4.0 |
| Hugging Face `tars-robotics/WIYH` | **已发布** — 全量约 **36.5 TB**（~1000 h）；样例 tar 可下 |
| 全量数据 | **已发布**（README TODO 勾选 Full Data，2025-04-10 口径） |
| Oracle Suite 硬件设计 | **宣称将开源** — 论文与 README 写「All data and hardware design will be open-source」；项目页侧重数据与 devkit |
| Foundation Model / Challenge | **待发布** — README TODO 未勾选 |

**结论：** **已开源（数据 + devkit）**；硬件 CAD/ BOM 与官方 foundation model 截至入库日未列独立仓库。

## 关联资料

- 论文：[`sources/papers/wiyh_arxiv_2512_24310.md`](../papers/wiyh_arxiv_2512_24310.md)
- 仓库：[`sources/repos/world-in-your-hands.md`](../repos/world-in-your-hands.md)
- 数据集：[`sources/datasets/wiyh.md`](../datasets/wiyh.md)
- Wiki：[`wiki/entities/paper-wiyh.md`](../../wiki/entities/paper-wiyh.md)
