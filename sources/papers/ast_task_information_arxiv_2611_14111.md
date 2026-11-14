# Model-Free Runtime Measurement of Task-Relevant Information｜AST 任务相关信息在线监测

- **类型：** paper
- **原文：** https://arxiv.org/abs/2611.14111
- **PDF：** https://arxiv.org/pdf/2611.14111
- **核对范围：** arXiv HTML v1 正文：任务信息定义、task binding、monitor、跨域实验及局限；附录 A–H 的领域接口、统计及贝叶斯收益率模型；对照官方嵌套仓 README 和 reproduce.sh。未运行源码。
- **通读：** 已完成
- **收录日期：** 2026-11-14
- **代码或数据：** https://github.com/dimipan/ast-telemetry

## 核心摘录

1. 领域接口把观测和目标转为可比较的任务解析状态。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ast-task-information.md)
2. 固定因果监测器输出 progress 和 friction 两个连续指标。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ast-task-information.md)
3. 可将 friction 用作运行诊断或 RL 惩罚，但需要领域任务接口。
   - **对 wiki 的映射：** [方法与评测](../../wiki/entities/paper-ast-task-information.md)

## 作者报告的结果与限制

作者在八类领域使用同一监控配置，覆盖网格探索、搜索救援、机械操作与意图估计；论文摘要报告对无效停滞的区分能力。

任务信息坐标需人为定义，假设已获得证据持续有效；无法代替实测成功率。

## 原文核对摘录（2026-11-14）

论文框架 Acquisition-State Telemetry（AST）先由领域 **task binding** 将观测投影为任务坐标的 resolution score 与单位范数 evidence embedding，再由固定 Monitor 输出 progress、friction、gains；无须拟合被监控的策略，不等于不需要任务 schema。附录的贝叶斯进度估计对各需求坐标的收益率赋 Beta(1,1) 先验，根据有收益次数更新，辅以历史证据相似度和检查点直行度评估无效重复。覆盖 SAR、MapTask、MultiWOZ、AnnoMI、MiniGrid、Crafter、MetaWorld、BOIR 及 RL 控制；MetaWorld 对动作前停滞与完成后停住的区分存在差异。**官方源码并非无可运行入口：** GitHub 的嵌套目录 `ast_telemetry_anonymous_repo/ast-telemetry/` 包含 `reproduce.sh`；`./reproduce.sh studies` 调用各 study1–3 脚本，`JOBS=16 ./reproduce.sh rl` 调用 study4 训练/评测，最终运行 figures.py 与 check.py。运行前必须设置 AST_DATA 指向独立数据集。

## 复现边界

尚未独立运行数据、训练、评测或机器人试验；论文数字为作者报告。对 wiki 的映射：[论文详情](../../wiki/entities/paper-ast-task-information.md)。
