# From Digital Human Interactions to Physics-Based Humanoid Skills: Physics-Grounded Post-Training of Interaction Generators

- **类型：** paper（预印本，v1）
- **论文：** <https://arxiv.org/abs/2610.10322v1>
- **项目页：** <https://1233chen.github.io/DIGHT/>
- **作者：** Kerui Chen、Jianrong Zhang、Kai Lv、Hehe Fan
- **机构：** 浙江大学、北京交通大学
- **入库日期：** 2026-11-05

## 核对状态

目前仅核对到论文摘要和项目页源码；论文全文在当前环境无法获取，尚未通读方法、实验及附录。以下为摘要层面的待核实线索，不作为完整论文解读。

## 摘要线索

- DIGHT 将数字人交互生成器和人形机器人跟踪策略结合，在仿真中执行文本条件生成的候选动作。
- 基于执行轨迹构建可执行性与交互保真度的偏好，采用物理解耦的 diffusion DPO 对齐生成器，再利用生成动作微调跟踪策略。
- 项目页提供交互视频。代码和数据的开放状态尚需进一步核实；项目页源码中 Code 按钮未提供有效链接。

## 待完成

- 获取并通读论文全文及相关附录，核实方法细节、实验、定量结果与局限。
- 核实项目代码和数据状态；完成论文实体页、基于原文的 Mermaid 流程图及交叉链接。
- 在可用完整仓库环境中执行 `make ci-preflight` 和必要检查。
