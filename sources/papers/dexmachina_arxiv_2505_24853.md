# DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation

> 来源归档（ingest · REALab 14 篇盘点）

- **标题：** DexMachina: Functional Retargeting for Bimanual Dexterous Manipulation
- **类型：** paper
- **状态：** ICML 2026
- **原始链接：**
  - arXiv：<https://arxiv.org/abs/2505.24853>
  - 项目页：https://project-dexmachina.github.io/
- **代码：** https://project-dexmachina.github.io/
- **作者：** Mandi Zhao, Yifan Hou, Dieter Fox, Yashraj Narang, Ajay Mandlekar, Shuran Song
- **机构：** Stanford University; NVIDIA
- **入库日期：** 2026-08-18
- **一句话说明：** DexMachina（ICML 2026）：VOC 课程+任务/运动/接触奖励的功能重定向；双手灵巧长时程 benchmark；仿真显著优于基线；真机鲁棒性待验。

## 核心论文摘录（MVP）

### 问题与贡献

- **摘录要点：** 从人类手–物示范学习机器人双手灵巧策略：虚拟物体控制器（VOC）课程 + 多奖励 RL，在仿真 benchmark 上做功能重定向。
- **对 wiki 的映射：**
  - [wiki/entities/paper-dexmachina.md](../../wiki/entities/paper-dexmachina.md)

### 方法与结果（归纳）

- **方法：** 从示范提取任务/运动/接触奖励；VOC 先驱动物体到目标再让策略接管；多灵巧手多任务 benchmark。
- **评测：** 仿真多任务显著优于 IK/位置引导等基线；支持跨硬件功能比较。

## 当前提炼状态

- [x] 公众号盘点 + arXiv/项目页交叉核对
- [x] wiki 实体页：`wiki/entities/paper-dexmachina.md`
