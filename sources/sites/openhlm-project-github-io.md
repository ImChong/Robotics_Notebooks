# OpenHLM 项目页（openhlm-project.github.io）

> 来源归档（ingest 配套站点）

- **URL：** <https://openhlm-project.github.io/>
- **备用/徽章链：** <https://openhlm-corl.github.io/>（README badge；内容与主项目页同源叙事）
- **标题：** OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation
- **类型：** project site
- **机构：** 清华大学；上海期智研究院；千寻智能（Spirit.AI）
- **论文：** <https://arxiv.org/abs/2606.22174>
- **代码：** <https://github.com/OpenHLM-project/OpenHLM>
- **数据集：** <https://huggingface.co/datasets/OpenHLM/OpenHLM-data>
- **权重：** <https://huggingface.co/OpenHLM/OpenHLM-ckpts>
- **入库日期：** 2026-07-22
- **一句话说明：** 官方落地页：全身原生人形 VLA「经验配方」——关节级全身遥操作数据采集、π₀.₅ 系 VLA 适配消融、HuMI/stationary 异构共训；长程果盘任务 **87.5%** progress（<½ 演示时长）优于 GR00T N1.6 / Ψ₀；12 项语言条件全身 loco-manip 任务平均 progress >90%。

## 开源状态（2026-07-22 项目页核查）

| 产物 | 状态 |
|------|------|
| 代码 | **已开源** · Apache-2.0 · [OpenHLM-project/OpenHLM](https://github.com/OpenHLM-project/OpenHLM) |
| 训练数据 | HF `OpenHLM/OpenHLM-data` |
| 检查点 | HF `OpenHLM/OpenHLM-ckpts` |
| 硬件/采集文档 | 仓内 `GR00T-WholeBodyControl-4-OpenHLM` + `HuMI4OpenHLM` |

## 页面要点（2026-07 快照）

### 三阶段经验路线图

1. **低层控制与遥操作接口：** 解耦控制（21-D）、VR 三点（24-D）、**关节级全身遥操作（32-D，采用）** —— 仅后者可表达踩踏板/蹲身穿架等全身自由度。
2. **全身 VLA 设计消融：** 动作投影/排序/相对动作/本体等接口适配影响小；**非人形机器人预训练（π₀.₅）** 远强于 PaliGemma / 随机初始化；**多步 flow** 优于 one-step。
3. **异构共训：** Stationary teleop 与 **HuMI**（无机器人、UMI 类可穿戴）可把 held-out 任务 progress 从 ~36% 抬到 ~84–89%，接近全全身遥操作 oracle（~96%）。

### 系统级对比（项目页）

| 系统 | 演示时长 | 长程果盘 Task Progress |
|------|----------|------------------------|
| **OpenHLM（HuMI 共训）** | **1.14 h** | **87.5%** |
| GR00T N1.6 | 2.70 h | 57.5% |
| Ψ₀ | 2.70 h | 48.8% |

### 十二任务四类

Pick & Place / Workspace Extension / Body-as-Tool / Constraints（Cola Placement、Shelf Cup Transfer、Bottle Disposal、Jar Opening、Cart Pushing、Pouring 等）。

## 对 wiki 的映射

- [OpenHLM 实体页](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
- [代码归档](../repos/openhlm.md)
- [论文摘录](../papers/openhlm_arxiv_2606_22174.md)
