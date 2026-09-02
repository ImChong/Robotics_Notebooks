# The Imitator Game（官方仓库）

> 来源归档

- **标题：** The Imitator Game
- **类型：** repo
- **链接：** <https://github.com/imitator-game/The-Imitator-Game>
- **站点仓：** <https://github.com/imitator-game/imitator-game.github.io>
- **论文：** <https://arxiv.org/abs/2608.22301>
- **项目页：** <https://imitator-game.github.io/>
- **许可：** MIT（Copyright 2026 Xunzhe Zhou）
- **入库日期：** 2026-09-02
- **一句话说明：** ManiSkill/SAPIEN 仿真框架 + IG-10K 采集/预处理 + 九套模仿基线的训练与评测 recipe。
- **沉淀到 wiki：** [`wiki/entities/paper-imitator-game.md`](../../wiki/entities/paper-imitator-game.md)

## 仓库入口（README，2026-09-02）

| 组件 | 路径 / 命令 |
|------|-------------|
| 安装 | `uv sync --active`；再按 ManiSkill 文档配 Vulkan；README 要求补丁一份 lerobot `lerobot_dataset.py` |
| 仿真资产 | `hf download imitator-game/IG-10K-Assets --repo-type dataset --local-dir ~/.maniskill/data` |
| 配对数据 | `hf download imitator-game/IG-10K-Dataset --repo-type dataset --local-dir demos` |
| 仿真采集 | `scripts/collect_data.py`；示例 `python -m mani_skill.examples.motionplanning.dual.two_robot_run` |
| h5 → LeRobot | `examples/baselines/lerobot_dataset/h5_to_lerobot.py` |
| 训练 | `python -m examples.baselines.<model>.train_model_imitator`（人/机 root + task mapping） |
| 评测 | `python -m examples.baselines.<model>.eval_model_imitator` |
| 任务模板 | `mani_skill/envs/tasks/_template/` |
| 模型模板 | `examples/baselines/_template/` |
| 本体模板 | `mani_skill/agents/robots/_template/` |

## 已发布基线目录

`examples/baselines/`：`act`、`diffusion_policy`、`vqbet`、`uniskill`、`xskill`、`gr00t`、`rdt`、`pi`、`openvla_oft`，另含 `encoders`、`lerobot_dataset`、`hand_estimation`。

## 边界

- **仿真与数据管线可跑**；真机部署评测由作者在其 Realman 平台代跑，需社区申请，仓内暂无策略上传接口。
- HF 未挂官方 checkpoint；复现数字需按各基线 README 自训。
- 仿真栈跟上游 ManiSkill 滚动更新，贡献必须走共享模板。

## 关联资料

- 论文：[`sources/papers/imitator_game_arxiv_2608_22301.md`](../papers/imitator_game_arxiv_2608_22301.md)
- 项目页：[`sources/sites/imitator-game-github-io.md`](../sites/imitator-game-github-io.md)
- 数据集：[`sources/datasets/ig-10k.md`](../datasets/ig-10k.md)
