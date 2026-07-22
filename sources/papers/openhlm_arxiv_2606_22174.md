# OpenHLM（arXiv:2606.22174）

> 来源归档（ingest）

- **标题：** OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation
- **缩写：** **OpenHLM**
- **类型：** paper / humanoid / loco-manipulation / VLA / teleoperation / co-training
- **arXiv：** <https://arxiv.org/abs/2606.22174>
- **项目页：** <https://openhlm-project.github.io/>
- **代码：** <https://github.com/OpenHLM-project/OpenHLM>
- **数据 / 权重：** HF `OpenHLM/OpenHLM-data` · `OpenHLM/OpenHLM-ckpts`
- **作者：** Yingdong Hu, Haodong Zhu, Boyuan Zheng, Yihang Hu, Tong Zhang, Zunhao Chen, Junming Zhao, Ruiqian Nai, Yang Gao
- **机构：** 清华大学；上海期智研究院；千寻智能（Spirit.AI）
- **发表日期：** 2026-06-20
- **入库日期：** 2026-07-22
- **开源状态：** **已开源**（代码 + 数据 + checkpoint；项目页与 README 互链）
- **一句话说明：** 用 **单变量消融路线图** 回答「如何做全身原生人形 VLA」：关节级全身遥操作优于部分自由度接口；静态/轮式双臂预训练可迁移到人形全身动作空间；**HuMI** 异构共训扩展新物体/指令而无需额外全身遥操作；系统级长程任务以更少演示时长超过 GR00T N1.6 与 Ψ₀。

## 核心论文摘录（MVP）

### 1) 问题与主张（Abstract）

- **痛点：** 多数人形系统把上下身解耦成「轮式双臂式」行为，无法协调整条运动链。
- **问法：** 构建 **语言+像素 → 全部自由度** 的 whole-body-native VLA，需要哪些经验决策？
- **答法：** 三阶段 one-variable-at-a-time：全身遥操作接口 → VLA 设计 → 异构共训 → 得到可复现 **OpenHLM recipe**。
- **对 wiki 的映射：**
  - [OpenHLM](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 遥操作接口（Phase 1）

- **对照：** Decoupled（21-D）/ VR 3-point（24-D）/ **Joint-based whole-body（32-D）**。
- **结论：** 仅关节全身接口能完成踩踏板、蹲身穿架等任务；后两阶段均建立在该接口上。
- **对 wiki 的映射：**
  - [Teleoperation](../../wiki/tasks/teleoperation.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 3) VLA 设计与异构共训（Phase 2–3）

- **接口适配**（投影初始化、动作排序、相对/绝对、本体）单点翻转仅轻微掉点——**非瓶颈**。
- **预训练：** π₀.₅ 初始化 ~91% task progress ≫ PaliGemma ~60% ≫ random ~42%。
- **推理：** 多步 flow matching 优于 one-step / drifting（尽管后者验证 MSE 可能更低）。
- **共训：** 在 Tasks 1–8 全身遥操作基础上，对 held-out 9–11 加入 **HuMI** 或 **stationary teleop**，progress 36% → 84%/89%（oracle 全全身遥操作 96%）。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) — 低层执行栈对照

### 4) 系统结果（项目页 / 论文）

- 长程果盘（低桌–中桌–高架、双手分拣、20 有序果对）：OpenHLM **87.5%** @ **1.14 h** demo vs GR00T N1.6 **57.5%** / Ψ₀ **48.8%** @ **2.70 h**。
- 12 任务平均 task progress **>90%**（四类：抓放+行走、工作空间扩展、身体当工具、环境约束）。

## 对 wiki 的映射（汇总）

- 主实体：[`wiki/entities/paper-loco-manip-161-154-openhlm.md`](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
- 站点 / 仓：[`sources/sites/openhlm-project-github-io.md`](../sites/openhlm-project-github-io.md)、[`sources/repos/openhlm.md`](../repos/openhlm.md)
- 161 策展索引：[`loco_manip_161_survey_154_openhlm.md`](./loco_manip_161_survey_154_openhlm.md)
