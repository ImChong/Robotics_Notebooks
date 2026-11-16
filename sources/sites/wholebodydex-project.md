# Whole-Body Dexterity 项目页归档

> 来源归档

- **标题：** Whole-Body Dexterity
- **类型：** site（项目页）
- **URL：** <https://www.wholebodydex.com/>
- **论文：** <https://arxiv.org/abs/2610.22184>
- **机构：** Amazon；加州大学圣地亚哥分校；密歇根大学
- **核查日期：** 2026-11-13
- **开源状态：** 2026-11-13 再核查项目页：提供 Paper、arXiv、BibTeX、实验指标与演示；未列官方 GitHub/Hugging Face 训练或推理代码、权重、数据集下载入口。属于**未提供公开可运行入口**，不等于已经证明作者不打算开源。
- **论文摘录：** [whole_body_dexterity_arxiv_2610_22184.md](../papers/whole_body_dexterity_arxiv_2610_22184.md)
- **沉淀到 wiki：** [paper-whole-body-dexterity.md](../../wiki/entities/paper-whole-body-dexterity.md)

## 页面核查摘要

项目页展示了篮球与纸箱的肩部重定位、腋下夹持及货架到桌面的长时程搬运，并给出汇总试验完成率：WBD 45/50、直接运动重定向 9/50、SONIC 2/50。页面提供遥操作设置和分阶段任务指标，但未发现可下载代码或数据集链接；不建立 `sources/repos/` 条目。

## 进一步核对（2026-11-13）

- 页面明确区分 WBD（45/50）、GMR（9/50）和 SONIC（2/50）的 **50 次真实试验汇总**，不能将 90% 用作无遥操作条件下的自主任务成功率。
- 试验由一位熟练操作员以 **Xsens 动捕服 → SMPL 身体动作命令** 引导上半身；下半身由单独控制器处理。
- 货架到桌面任务的 WBD 最终放置成功 **7/10**；按任务阶段统计依次为拾取 10/10、腋下夹持 9/10、搬运 9/10、解夹 9/10、放置 7/10。
- 页面没有展示可运行源码入口，故无需建立不存在的 `sources/repos/` 条目；若后续发布，再根据真实 README/源码补运行时序图。
