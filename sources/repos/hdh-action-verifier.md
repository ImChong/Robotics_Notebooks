# Learning Low-Latency Action Verification for Frozen Vision-Language-Action Models — 官方代码仓库

- **官方仓库：** https://github.com/Wen-HanHsieh/hdh_real
- **论文：** https://arxiv.org/abs/2611.14834
- **项目：** https://wen-hanhsieh.github.io/hdh/
- **开放结论：** 已公开 Franka 真机代码；另外有 hdh_sim 仿真仓库。
- **核对范围：** 仓库首页 README、项目开放标识；真实源码运行入口和调用链仍待逐项核实
- **通读：** 未完成

## 研究对应关系

冻结 π0.5 收集自主 rollouts，按回合终止成败标注当前观测与执行动作前缀；训练带历史条件的轻量 verifier；推理时采样两个候选块，按成功分数选择，并执行获选块的前 8 步。

## 复现说明

当前未声明已实际运行。先按仓库 README 核对模型、依赖、数据和入口，再补运行时序图。

## 对 wiki 的映射

- [论文详情](../../wiki/entities/paper-hdh-action-verifier.md)
