# COWBOY（官方项目页）

> 来源归档（ingest）

- **标题：** COWBOY: A Scalable Sim-to-Real Framework for Learning Contextual Whole-Body Manipulation
- **类型：** site / project-page
- **URL：** <https://cowboy-manip.com/>
- **论文：** <https://arxiv.org/abs/2610.28948>
- **会议：** Conference on Robot Learning（CoRL）2026
- **作者：** Jiahui Yang、Jason Jingzhou Liu、Ray Muxin Liu、Andrew E. Wang、Kenneth Shaw、Deepak Pathak、Ruslan Salakhutdinov
- **机构：** Carnegie Mellon University（CMU；项目页展示 CMU 标识）
- **代码：** 项目页截至 2026-10-23 未列出 GitHub 或其他代码仓库
- **入库日期：** 2026-10-23
- **论文归档：** [COWBOY arXiv 摘录](../papers/cowboy_arxiv_2610_28948.md)
- **Wiki：** [COWBOY 实体页](../../wiki/entities/paper-cowboy-contextual-whole-body-manipulation.md)

## 一句话摘要

COWBOY 将环境感知全身控制、局部接触强化学习和多教师蒸馏组合起来，训练一个视觉策略，让移动操作机器人根据场景几何调整接近路线、主动视角和灵巧抓取动作。

## 项目页核查（2026-10-23）

- 项目页明确标注 CoRL 2026，提供论文 PDF、摘要、方法说明及多场景演示。
- 页面对真机评测的表述为：24 个真实场景、100 多种独特物体、91% 抓取成功率；这是项目页报告的整体结果，页面未在该数字旁给出逐场景明细。
- 单一视觉策略协调 **32 个自由度**：移动底座 3、手臂 7、灵巧手 16、主动视觉颈部 6。
- 方法由三部分组成：GPU 向量化环境感知 WBC、接触丰富的局部 RL 策略、从局部场景到完整场景的多教师蒸馏。项目页称 WBC 可在单 GPU 上覆盖数千仿真环境。
- 项目页介绍 SAM3 可根据语言提示分割目标物体，再由估计位置引导同一策略完成任务。
- **开放状态：** 页面当前列出的链接未包括源码仓库、权重或数据集下载入口；本次不建立官方仓库归档，也不把“页面未列出链接”写成作者承诺开源或永久未开源。
