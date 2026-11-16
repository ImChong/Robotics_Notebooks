# GazeVLA: Learning Human Intention for Robotic Manipulation / GazeVLA：从人类注视学习操作意图

> 来源归档（论文全文精读，非转载）

- **类型：** paper / vision-language-action / gaze / human-to-robot-transfer / manipulation
- **论文：** https://arxiv.org/abs/2604.22615
- **全文：** https://arxiv.org/pdf/2604.22615v3
- **版本：** arXiv v3，2026-11-02（v1 2026-04-24）
- **会议：** CoRL 2026，Spotlight（作者主页、公开会议 Spotlight 索引列示）
- **作者：** Chengyang Li、Kaiyi Xiong、Yuan Xu、Lei Qian、Yizhou Wang、Wentao Zhu
- **单位：** 上海交通大学、宁波东方理工大学、北京大学、上海科技大学
- **项目页：** https://gazevla.github.io/
- **官方代码：** https://github.com/lichy2004/GazeVLA
- **官方数据处理：** https://github.com/lichy2004/GazeVLA-Data
- **核对范围：** 通读 arXiv v3 的摘要、引言、相关工作、§3 数据/架构/训练、§4 仿真与真机及消融、§5 局限和补充材料 A–E（数据对齐、模型设置、实验协议、OOD 与意图源）；按表 1–6 和图 1–15 核对论证与指标。参考文献用于定位相关工作，不将其作为本文直接证据。
- **通读：** 已完成
- **入库：** 2026-11-08

## 核心问题

如何让人类第一视角演示对机器人操作有用，而不要求人–机器人具有一致的关节/动作空间？论文将**注视点（gaze）作为意图的二维代理变量**，让 VLA 在视觉语言条件下先推断意图，再输出连续动作块。

## 数据与建模

- 数据：聚合 **13 个第一视角数据源，超过 150M 帧**；统一手部与注视点坐标、有效性 mask；用 Qwen 辅助切分长视频，剔除无手–物交互片段、眼跳/低可信目光。
- 人体动作：基于 MANO / 手部关键点整理为 **48 维**手指端点与手腕位置/旋转表示；注视意图为图像坐标 (h,w) 的 **2 维**点。
- 模型：论文正文把内部策略称 **VLIA**；基于 PaliGemma-2（正文描述 PaliGemma，附录细化为 3B 版）+ SigLIP 视觉编码 + flow-matching action expert。
- 因式分解：p(a, i | s, o, l) = p(i | o, l) × p(a | i, s, o, l)。视觉语言骨干自回归预测离散 gaze tokens；动作专家在上述中间表示条件下生成连续动作块。
- 训练：先人类数据大规模预训练（先冻结 VLM warm-up，随后联合优化），再用小规模人类与机器人示范以 **1:1** 比例后训练；**机器人侧不需要 gaze 标签**。损失包括意图 next-token 与动作 conditional flow-matching。
- 细节：附录使用 224×224 图像、224 个轴向 bins 量化 2D gaze、跨图像与 gaze 的同步空间增广；预训练 20k step / 8×A800 / 1,344 GPU h，后训练 30k step（论文表 4）。

## 论文报告的结果（作者实验，非独立复现）

| 设置 | GazeVLA/VLIA | 参照与说明 |
|---|---:|---|
| AV-ALOHA，六任务 ID 平均成功率 | 49% | π0.5 41%、LFA 43%、HRDT 39%；每任务 100 条示范、100 次测试 |
| AV-ALOHA，OOD 干扰物平均 | 28% | π0.5 22% |
| AV-ALOHA，OOD 光照平均 | 27% | π0.5 23%；不是每个任务均领先 |
| 真机六任务平均提升 | +12.1 个百分点 | 相对 π0.5；ALOHA 与 Unitree G1，包括长程/精细/灵巧任务 |
| 无人类预训练 → 完整模型，OOD-Scene | 0/10 → 6/10 | Pick-and-place 消融，样本很少 |
| 无意图 → 完整模型，ID | 16/20 → 19/20 | Pick-and-place 消融 |
| Fruit Collecting 意图源 | 预测 gaze 65%；人类 gaze 65% | VLM keypoint 50%；无意图 40%；随机意图 25% |

- 论文还测得 224×224 条件下人类注视预测平均误差约 **11 px**（图像对角线的 4.8%）。
- 对意图 token 的人为干预会改变所抓目标；**不意味着机器人部署时必须佩戴人类眼动仪**。
- 仿真中动作 chunk 为 25 步、25 FPS（1 秒），执行时开环；部署实时性需单独评估。

## 局限与核验边界

- 论文 §5 明确指出：先意图后动作的推理链增加额外计算开销；仍依赖机器人后训练，**不是完全 zero-shot**。
- 真实实验任务与示范预算有限，不能将桌面操作的结果直接外推到全身移动操作。
- 数值均为作者报告；官方 README 截至 2026-11-08 仍把 checkpoint 与评测脚本标为 TODO，且训练入口存在模块命名不一致（见源码归档），不将本文数字标作独立复现。
- 代码 README 与仓库 LICENSE 对许可表述不一致，复用前须核实（见源码归档）。

## 映射

- [唯一论文项目实体](../../wiki/entities/paper-gazevla.md)
- [官方项目页归档](../sites/gazevla_project.md)
- [官方代码归档](../repos/gazevla_code.md)
- [官方数据处理归档](../repos/gazevla_data.md)
