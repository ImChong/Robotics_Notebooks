# CarryOn: A Dual-Loop Harness for Personalized and Reliable Robot Assistance

> 来源归档（ingest），来自 2026-11-13「具身智能小站」18 篇论文阅读清单；以原始 arXiv 及官方项目和仓库为核查依据。

## 元数据

- **标题：** CarryOn: A Dual-Loop Harness for Personalized and Reliable Robot Assistance
- **中文：** CarryOn：具备记忆、安全与失败恢复的双循环机器人 Harness
- **arXiv：** 2610.21453
- **版本：** 2026-11-14 检查 arXiv HTML；具体修订号待逐项补齐
- **入库日期：** 2026-11-14
- **核对范围：** 已直接阅读 arXiv HTML 摘要、方法与部分实验/局限章节；尚未逐行检查全文及全部附录，后续继续核对
- **通读：** 未完成
- **论文：** https://arxiv.org/abs/2610.21453
- **HTML：** https://arxiv.org/html/2610.21453
- **PDF：** https://arxiv.org/pdf/2610.21453
- **官方项目页：** https://vincent2311.github.io/Projects/CarryOn/
- **仓库：** https://github.com/JIA-Lab-research/CarryOn

## 论文机制

规划循环基于用户偏好、场景观察与对话更新记忆，输出子任务；执行循环监视策略动作，对高风险接触进行主动 shielding，执行失败时调用控制式 recovery skill，复盘后修订下一版技能。

## 实验与证据边界

论文分别评估个性化、接触安全与恢复；在对应设置中，违反率从 52.6% 降至 6.1%，恢复成功率从 13% 升至 80%。各指标属于不同实验，不能合称端到端成功率。

## 复现与开放状态

官方仓库 README 标注 2026-10-25 已发布代码；本次无法直接加载项目页，开源状态以仓库 README 为依据。

## 局限

安全屏蔽仍依赖风险估计和接触建模；系统增益来自外部规划、屏蔽与恢复机制，不表示冻结 VLA 权重本身改善。

## 对 wiki 的映射

- [CarryOn：具备记忆、安全与失败恢复的双循环机器人 Harness](../../wiki/entities/paper-carryon-dual-loop.md)

## 参考来源

- [原论文 HTML](https://arxiv.org/html/2610.21453)
- [转载阅读线索（仅作索引）](https://mp.weixin.qq.com/s/gatO-d6IxsemBSk62EvS1w)
