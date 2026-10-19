# OneWM（官方项目页）

> 来源归档（ingest）

- **标题：** Can World and Action Modeling Benefit Each Other under Unified Causal Training?
- **类型：** site / project-page
- **URL：** <https://selen-suyue.github.io/OneWMNet/>
- **论文：** <https://arxiv.org/abs/2610.21095>
- **代码：** <https://github.com/Interactive-Active-AI/OneWM> — [仓库归档](../repos/onewm.md)
- **作者与机构：** Yue Su 等；MMLab@HKU、MVIG@SJTU、THU、SII、UCAS、IAAI Team
- **入库日期：** 2026-10-19
- **论文归档：** [OneWM arXiv 摘录](../papers/onewm_arxiv_2610_21095.md)

## 一句话摘要

OneWM 项目页展示一个因果视频—动作模型如何在共同架构中学习动作条件世界预测和动作生成，并用仿真、真机操作与想象 rollout 演示其能力。

## 开源状态核查（2026-10-19）

- **代码：已公开。** 项目页 Code 链接到官方 GitHub；仓库包含 LIBERO、RoboCasa365、dual Franka 的训练配置、训练入口和部署/评估代码。
- **许可证：Apache-2.0。** 官方仓库根目录提供 LICENSE，并在 README 声明代码许可。
- **权重：未随仓库提供。** README 的推理示例要求调用者提供训练好的 checkpoint；本次项目页未发现公开权重下载入口。
- **数据：需另行准备。** README 指定 LIBERO LeRobot v2.1 数据、RoboCasa365 Human300 预训练数据及 Franka LeRobot v3 录制数据；仓库不随附这些数据，真机录制需用户提供。
- **想象演示边界：** 项目页说明 16 轮想象 rollout 不使用之后的真实观测或记录动作；该演示评估想象中的任务推进，未在模拟器执行预测动作，也没有自动成功检测。

## 关联资料

- 论文：[sources/papers/onewm_arxiv_2610_21095.md](../papers/onewm_arxiv_2610_21095.md)
- 代码：[sources/repos/onewm.md](../repos/onewm.md)
- wiki：[wiki/entities/paper-onewm-unified-causal-training.md](../../wiki/entities/paper-onewm-unified-causal-training.md)
