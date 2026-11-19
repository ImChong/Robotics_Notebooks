# World Models as Unit Tests / 把世界模型当作单元测试

> 一手研究笔记（research note），不作为经同行评审的论文或真机实验记录。

- **标题：** World Models as Unit Tests
- **中文标题：** 把世界模型当作单元测试：Agent 用少量观测试验编写物理环境的可执行验证器
- **作者：** Denzel Zheng（Dongzhe Zheng）
- **机构：** 普林斯顿大学机械与航空航天工程系（Princeton University MAE）
- **时间：** 2026-11（网页署名）
- **类型：** 官方研究笔记 / 项目主页
- **原文：** https://dongzhezheng.github.io/world-models-as-unit-tests/
- **代码：** https://github.com/DongzheZheng/world-models-as-unit-tests
- **数据：** https://huggingface.co/datasets/DenzelZheng/world-models-as-unit-tests
- **交互回放：** https://huggingface.co/spaces/DenzelZheng/world-models-as-unit-tests
- **入库：** 2026-11-19
- **开源状态：** 已开源，官方仓库 MIT 许可；仿真环境、世界模型、控制程序、记录、复现与 harness 代码公开。原始 agent 生成过程不能由仓库直接重跑，harness 尚未由真实模型完整实测。
- **核对范围：** 作者网页全文 §1–§10（含任务定义、系统辨识、34 版程序对比、real-to-sim-to-real、Cable 与 Rope 的反例、模型大小、实验限制、相关工作、总结），核对官方 README、harness/README.md、reproduce.py 的 CLI 与可复现协议。
- **通读：** 已完成

## 原始材料核心证据

1. **研究问题：** Agent 编写机器人控制代码的速度超过真实试验速度；尝试让 agent 在有限试验里主动观察物理环境、识别隐藏参数，并写出 14–26 KB Python 可执行世界模型，用它对不同控制程序做随机化模拟验收（网页 §1–§2、§7）。
2. **实验边界：** 8 个任务、4 类模拟器（VizDoom、PyBullet/PyFlyt、pymunk、MuJoCo）；各 task 的初始观测预算为最多 5 次“real trials”，这里 **real 指隔离的高保真原始模拟器而非实体机器人**（网页 §1、§9）。
3. **评测：** 34 个 CaP 程序在五类任务上的世界模型预测与原模拟器成功率 Pearson r=0.974、MAE=5.2 个百分点；这些数字是作者在特定版本程序上的结果，且部分物理任务预测系统性偏乐观（网页 §3、§9）。
4. **策略选型：** 无人机 mission C 中，CaP 的 20 次试验版本 76%，再加世界模型测试、没有新增原环境试验后 95%；Push-L 从 46% 到 84% held-out，重测 80%（网页 §4）。这些对照不应误说成整个过程仅需 5 次试验。
5. **在线规划：** Rope 抛投任务里基于短时预测的 MPC 每 0.16 s 重新规划、6 个扰动世界模型筛选候选动作；20 次试验版本在原模拟器中为 78%，oracle 84%，但约 2 s 计算才能执行 0.02 s 物理步，不是实时部署能力（网页 §6、§9）。
6. **不适用/风险：** Cable 有实时载荷反馈时 CaP 已达到 50/50，该任务世界模型不增加成功率，只发现罕见路径 bug；作者明确表示所有任务都是模拟器、没有真机 sim-to-real 测试，而且每个 agent/task 仅一次运行（网页 §5、§9）。

## 研究判读

- 世界模型的主要价值是 **给代码策略提供可重复、便宜的验收环境**，不是证明模型有绝对可信的动力学。
- 严格区分“世界模型中的预测成功率”“原环境 held-out 成功率”和“真机成功率”；第三项在本研究未测。
- 公开研究属于 **research note** 而非正式审稿论文。

## 对 wiki 的映射

- [独立项目实体：World Models as Unit Tests](../../wiki/entities/world-models-as-unit-tests.md)
- [官方源码归档](../repos/world-models-as-unit-tests.md)
