# Ai2 ToyBlocks: A controlled world for robot learning and evaluation

- **类型：** 官方博客 / 官方技术文档
- **中文标题：** ToyBlocks：可控的机器人学习与评测基础世界
- **作者机构：** Allen Institute for AI（Ai2）
- **发布时间：** 2026-11-06
- **原文：** https://huggingface.co/blog/allenai/toyblocks
- **官方项目：** https://allenai.github.io/ToyBlocks
- **官方代码：** https://github.com/allenai/ToyBlocks
- **代码：** https://github.com/allenai/ToyBlocks
- **数据集：** https://huggingface.co/datasets/allenai/toyblocks-data
- **模型：** https://huggingface.co/collections/allenai/toyblocks
- **官方文档：** https://allenai.github.io/ToyBlocks/docs/user_guide/index.html
- **检查日期：** 2026-11-11
- **核对范围：** 通读 2026-11-06 Ai2 官方博客（机制、VLA 泛化、VLM、MCP 智能体、世界模型、FR3 Sim2Real、开放计划）全部正文；核对项目 Overview、Experiments、Sim-to-real 页面及 GitHub README 和数据/评测入口；未下载完整数据、运行 GPU 训练或真机复现。
- **通读：** 已完成

## 原文要点
1. Isaac Sim/Isaac Lab 的可控桌面积木场景，用 PDDL 任务规格生成可规划的积木推、翻、堆叠操作。
2. 自动演示流水线包含 task generation → 环境与重置 → FF/PDDL 规划与技能执行 → 可选分离渲染 → LeRobot 合并与标注，不依赖人工遥操作逐条采集。
3. 作者报告约 85,336 条演示、近 1,600 小时、约 1.44 亿帧；这些是特定发布数据集口径，不是每个默认单 GPU quick start 会生成的规模。
4. 低层 ManiFlow 生成具身动作；高层可用 Molmo2 等视觉语言模型决策技能。仿真基准可隔离高低层 oracle 定位失败。
5. 论文/博文报告的仿真和物理机器人成功率分开统计；当前完整的 RL 训练集成仍列作未来工作，不能将演示数据开放等同 RL pipeline 全部可复现。

## 深度核验补充（2026-11-11）

### 数据与方法

- 首页口径：85,336 条长时序演示、约 1,600 h、144M 帧；约 30K 训练问题、600 个验证问题，481,207 个 skill steps、每段中位数 5 skills。首发数据无需逐条遥操作，**不等于全部短技能没有 RL 训练**。
- PDDL/TAMP 给出子任务与可执行技能；Isaac Lab 中通过 RL 学习的短技能与运动规划共同生成演示；高层输出动作/目标像素、底层使用视觉与目标热图控制。
- VLA 实验以四任务、三个模型（ManiFlow、π0.5、MolmoAct 2）在不同物体位置/堆叠高度上验证插值和外推；具体 Co-training、Mid-training、Delta actions、WAM、LoRA 单项设置及汇总见[官方实验完整页面](https://allenai.github.io/ToyBlocks/experiments.html)。
- 世界模型以四类技能、每类 900 条轨迹、成功失败各半测试位姿与结果预测；Wan 2.2 / Astra / LeWM 的指标属于**仿真真值参照**，不能替代真机接触验证。
- 智能体实验让模型通过 MCP 技能调用并自主判断成功/不可行；63% 包含**正确声明任务无法完成**，并非仅完成积木拼写的成功率。

### 真机与可复现性

- 原子技能仿真/真机成功率 **80.0% / 81.1%**（按技能命令）；完整 3 字母任务 **111/200（55.5%） / 15/30（50%）**（按 episode）；不可合并不同单位和分母。
- 实机 **Franka FR3 + Robotiq**、前方 ZED + 腕部 ZED Mini；对关节摩擦和 armature 做系统辨识、对物理和视觉做域随机化；通过 FastFoundationStereo 生成深度。
- [官方 Sim-to-real 页面](https://allenai.github.io/ToyBlocks/sim2real.html) 明确「policy-on-real-robot 代码及 system identification 指南」**Coming soon**；GitHub README 列独立技术报告 *coming soon*；博客同时明确完整 RL 训练能力尚未集成。已发布模拟器/策略与尚未发布的真机工具包应分开记录。

### 一手资料交叉链接

- [官方完整原始博文](https://huggingface.co/blog/allenai/toyblocks)
- [官方 Experiments](https://allenai.github.io/ToyBlocks/experiments.html)
- [官方 Sim-to-real Toolkit](https://allenai.github.io/ToyBlocks/sim2real.html)
- [官方项目首页](https://allenai.github.io/ToyBlocks/)
- [官方开源仓库](https://github.com/allenai/ToyBlocks)

## 对 wiki 的映射
- [ToyBlocks](../../wiki/entities/toyblocks.md)
- [源码归档](../repos/allenai_toyblocks.md)
