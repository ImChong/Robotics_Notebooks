# GE-Sim 2.0（AgibotTech 官方仓库）

> 来源归档

- **标题：** Genie Envisioner World Simulator 2.0 (GE-Sim 2.0)
- **类型：** repo
- **组织：** AgiBot / AgibotTech
- **代码：** <https://github.com/AgibotTech/GE-Sim-V2>
- **权重：** <https://huggingface.co/agibot-world/Genie-Envisioner-Sim-v2.0>
- **论文：** <https://arxiv.org/abs/2605.27491>
- **项目页：** <https://ge-sim-v2.github.io/>
- **入库日期：** 2026-05-30（2026-11-21 复核开源快照 `508c60f`）
- **核对范围：** 通读 `AgibotTech/GE-Sim-V2@508c60f` 的 `README.md`、`docs/worldoracle.md`、`scripts/serve_world_model.sh`、`examples/closed_loop.py`、`src/gesim/env.py`、`src/gesim/server/app.py`、`src/gesim/models/gesim_v2/model.py`；未运行 GPU 服务端或下载模型权重。
- **通读：** 已完成
- **一句话说明：** GE-Sim 2.0 官方闭环视频世界模拟器仓库：已于 2026-06-25 开源核心服务端/客户端与 G01+OmniPicker 蒸馏 4 步模型，并于 2026-10-17 增补 WorldOracle（G02/YAM）推理脚本与权重；World Judge 奖励模型权重未内置（提供可插拔 `RewardClient` 接口）。

---

## 仓库状态（README 与代码核查，2026-11-21）

| 项 | 状态 |
|----|------|
| 技术报告 | 已发布（arXiv:2605.27491，2026-05-28） |
| 闭环仿真代码（`src/gesim`） | **已开源**（2026-06-25 发布：`WorldModelEnv`、`WorldModelServer`、`_g01_fk.so`） |
| 模型权重（HF `Genie-Envisioner-Sim-v2.0`） | **已开源**（`gesim_community_v2.0.1_g01op_distill_2B`、`g02_2B`、`yam_2B`、`pi05_gesim_g01op_test`） |
| WorldOracle 评测入口 | **已开源**（2026-10-17 更新：`scripts/infer_worldoracle.py`） |
| World Judge 奖励权重 | **未随仓内置**（README 明确 `no reward model is bundled`，需用户自接 `RewardClient`） |

## 许可与上游

- 改编自 [Diffusers](https://github.com/huggingface/diffusers)、[Cosmos](https://github.com/nvidia-cosmos) 与子模块 `third_party/openpi` 遵循 **Apache-2.0**（openpi 基于 PaliGemma 另受 Gemma Terms of Use 约束）。
- 本仓其余数据与代码：**CC BY-NC-SA 4.0**（以 README 为准）。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 2B 动作条件多视角视频世界模型 + 闭环模块 |
| [Video-as-Simulation](../../wiki/concepts/video-as-simulation.md) | 像素级操纵仿真与闭环状态回传 |
| [EWMBench](../../wiki/entities/ewmbench.md) | 同 Agibot 生态：EWMBench 偏 **开环生成质量** 三维量；GE-Sim 2.0 偏 **闭环 rollout + 奖励接口** |
| [robot-world-models-training-loop-taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) | ③ 机器人视频世界模型 · 训练闭环案例 |

## 对 wiki 的映射

- 主实体页：**`wiki/entities/ge-sim-2.md`**
- 论文摘录：**`sources/papers/ge_sim_2_arxiv_2605_27491.md`**
- 项目页：**`sources/sites/ge-sim-v2-project.md`**
