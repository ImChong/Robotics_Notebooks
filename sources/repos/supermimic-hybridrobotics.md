# HybridRobotics/supermimic：SuperMimic 研究团队开源实现

> 来源归档（repo）

- **标题：** SuperMimic（此前 README 描述为基于 mjlab 的 SuperTrack 实现）
- **类型：** repo（研究团队公开的同名实现）
- **链接：** <https://github.com/HybridRobotics/supermimic>
- **论文：** <https://arxiv.org/abs/2610.28480>（[论文归档](../papers/supermimic_arxiv_2610_28480.md)）
- **项目页：** <https://supermimic.github.io/>（[项目页归档](../sites/supermimic-project.md)）
- **许可证：** Apache-2.0（以代码仓库的 LICENSE 为准）
- **入库日期：** 2026-11-11
- **核对范围：** GitHub README、LICENSE、`src/supermimic/scripts/train.py`、`play.py`、`train_utils.py`、`tasks/__init__.py`、`runners/supermimic_runner.py`、`learners/wm_learner.py`、`learners/policy_learner.py`、`envs/world_model_env.py` 中的训练、回放与世界模型调用路径；未运行训练、仿真或真机测试。
- **通读：** 已完成
- **一句话说明：** 基于 mjlab 的人形动作跟踪世界模型训练实现，源码提供 Unitree G1 的 GenCoord/MaxCoord 两种任务、训练及回放入口。
- **沉淀到 wiki：** [SuperMimic 统一论文实体](../../wiki/entities/paper-supermimic.md)

## 仓库身份与开放边界

HybridRobotics 组织发布同名代码，当前 README 明确说明基于 mjlab，并链接论文和项目页。训练、回放和基准入口已经公开；但截至 2026-11-11，论文项目页未直接链接此仓库。**不能仅凭仓库名和研究组织关联，声称它逐项复现了 CoRL 论文报告的全部指标或开放了论文所有训练资产及真机部署栈。**

## 经源码核对的入口

| 文件 | 实际职责 |
|---|---|
| `src/supermimic/tasks/__init__.py` | 注册 `SuperMimic-GenCoord-G1` 和 `SuperMimic-MaxCoord-G1` |
| `src/supermimic/scripts/train.py` | 解析任务与参数、解析本地或 W&B 动作资产、创建环境、初始化 `SuperMimicRunner` |
| `src/supermimic/runners/supermimic_runner.py` | 仿真采样、ReplayBuffer 写入、世界模型和策略交替更新、日志及 checkpoint |
| `src/supermimic/learners/wm_learner.py` | 从缓存采样的状态/动作序列拟合动力学，使用分通道加权状态预测损失 |
| `src/supermimic/learners/policy_learner.py` | 冻结世界模型，使用 `WorldModelEnv` 想象 rollout 反传策略；Actor-Critic 变体有 critic bootstrap 与 GAE |
| `src/supermimic/scripts/play.py` | 从 W&B 下载 `actor.pt`、`world_model.pt` 等，选择 native/Viser viewer 回放 |

示例训练命令（按 README 入口；需先安装依赖并准备动作资产）：

```bash
uv run src/supermimic/scripts/train.py SuperMimic-GenCoord-G1 --runner.run_name example
```

上述命令仅用于展示入口，不表示已执行。默认 `TrainConfig.num_envs=4096`，`WorldModelLearnerCfg.horizon=8`，策略 `ActorCriticPolicyLearnerCfg.horizon=32`；这些是当前源码默认配置，**不能直接等同于论文所有报告实验的最终配置**。

## 机制与限制

训练时 `SuperMimicRunner.run` 先通过真实 mjlab 环境 `step()` 采样，写入 `ReplayBuffer`，再在达到回放条件时调用 `WorldModelLearner.update` 与 `policy_learner.update`；后者通过可微 `WorldModelEnv.step` 计算奖励和终止概率。模型可存储为 `world_model.pt`、`actor.pt`、`critic.pt`，特定满足条件的模型才导出 ONNX。

- README 的动作数据依赖 W&B Registry 或显式 `--motion-file`；原论文全部数据、官方预训练权重与实机系统并非仓库代码目录即自动具备。
- 尚未在本次 ingest 中执行环境安装、GPU 训练、定量评测或 Sim2Real；论文指标仍归于论文作者报告。
- 代码路径核查记录仅覆盖上列核心文件和 README；其它模块未作逐行安全审计。

## 公开实现回访（2026-11-26）

- 默认分支固定提交 [`5fb9229`](https://github.com/HybridRobotics/supermimic/commit/5fb92297ab4fe3d86629d9f0c635334791bb0d5f) 的提交者时间为 2026-11-21 08:54:02 UTC（北京时间 **16:54:02**），作者时间为 2026-11-19 06:12:24 UTC。此处分别记两种 Git 时间，不把页面相对时间当版本证据。
- `_reanchor_targets` 用预测根与参考根之间的偏航差旋转目标刚体线速度和角速度，并对齐位姿目标；全局水平位置和航向由两个 anchor 奖励项单独计分，原有权重及尺度不变。新增测试覆盖单位旋转与非零偏航；本次仅静态核对。
- [`337c8b0`](https://github.com/HybridRobotics/supermimic/commit/337c8b043feee7b95cac363be4782c046ed17fa3) 将想象起点采样交由 Runner，再调用 `policy_learner.update(starts)`；环境不再持有回放缓存。[`d88804a`](https://github.com/HybridRobotics/supermimic/commit/d88804a872c94ac50031f98de4434d1684ec0ae4) 在 CUDA 上对 Dynamics MLP 和想象时 actor 局部使用 BF16，返回 FP32；不要将其实现更新等同于论文 H100 耗时测量。README 与 `pyproject.toml` 要求并列的 `../mjlab_research` 源码依赖。
- Apache-2.0 针对本代码，论文所用动作资产、权重和第三方依赖分别核对。未运行训练或 Unitree G1。

## 对 wiki 的映射

- [SuperMimic：可微世界模型与人形全身控制](../../wiki/entities/paper-supermimic.md) — 统一归档论文、项目页和本次研究团队代码。
- [CoRL 2026 官方议程](../sites/corl-2026-program.md) — 会议直播议程入口，不能替代逐场录像直接链接。
