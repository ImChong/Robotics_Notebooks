# BEHAVIOR Robot Suite 论文来源归档

## 元数据
- **题目：** BEHAVIOR Robot Suite: Streamlining Real-World Whole-Body Manipulation for Everyday Household Activities
- **作者：** Yunfan Jiang、Ruohan Zhang、Josiah Wong、Chen Wang、Yanjie Ze、Hang Yin、Cem Gokmen、Shuran Song、Jiajun Wu、Li Fei-Fei
- **机构：** Stanford University；项目与 Galaxea Dynamics 的 R1 机器人平台合作
- **会议：** Conference on Robot Learning (CoRL) 2025
- **预印本：** https://arxiv.org/abs/2503.05652
- **项目页：** https://behavior-robot-suite.github.io/
- **算法代码：** https://github.com/behavior-robot-suite/brs-algo
- **硬件控制代码：** https://github.com/behavior-robot-suite/brs-ctrl
- **数据：** https://huggingface.co/datasets/behavior-robot-suite/data
- **入库日期：** 2026-10-15
- **一句话说明：** BRS 将低成本全身遥操作 JoyLo、轮式双臂机器人和层级动作建模策略 WB-VIMA 组合为家庭长时程移动操作系统。

## 论文要点
BRS 使用 Galaxea R1 轮式双臂机器人与 4-DoF 躯干，面向家庭任务中的双臂协调、稳定导航和末端可达性。JoyLo 用 Joy-Con 与低成本双臂连杆采集全身示范；WB-VIMA 利用本体运动学层级，按上游到下游顺序自回归解码全身动作，并用 self-attention 汇集多模态观测。

项目页报告五项真实家庭任务评测，覆盖置物架、厕所清洁、整理衣物、派对后清理和把垃圾拿到室外。

## 对 wiki 的映射
- [BRS 实体页](../../wiki/entities/behavior-robot-suite.md)：硬件、JoyLo、WB-VIMA、任务评测、开源范围与复现注意点。
- [Loco-Manipulation 任务页](../../wiki/tasks/loco-manipulation.md)：用于区分轮式 mobile manipulation 与腿式 loco-manipulation。

## 开源状态
项目页分别链接算法代码、机器人控制代码与训练数据；官方 GitHub 仓库公开，数据集托管在 Hugging Face。BRS 可按完整系统入口复现，但硬件实现针对 Galaxea R1，迁移到其他平台需核查控制接口、标定与机构适配。

## 备注
该研究已在用户的 Robot Learning Paper Notebooks 中有独立论文深读。此处只归档项目身份和本知识库所需的系统级总结，不复制深读正文。
