# WCM（World Critic Model 官方实现）

> 来源归档

- **标题：** WCM — A History-Aware Critic for Partially Observable Robot Control
- **类型：** repo
- **来源：** 同济大学 / 上海创智学院 / 复旦大学（GitHub: `sylvestf`）
- **链接：** <https://github.com/sylvestf/WCM>
- **论文：** <https://arxiv.org/abs/2607.29613>（Submitted 2026-07-31）
- **项目页：** <https://sylvestf.github.io/wcm-homepage/>
- **权重 / 数据：** <https://huggingface.co/collections/Sylvest/wcm>
- **许可：** MIT
- **入库日期：** 2026-08-04
- **一句话说明：** 把「价值估计 + 动力学预测」合到一个轻量 LeJEPA critic 里的官方实现；四个编号 shell 脚本串起数据预处理 → 训练 → 评测 → 价值曲线可视化。
- **沉淀到 wiki：** [`wiki/entities/paper-wcm-world-critic-model.md`](../../wiki/entities/paper-wcm-world-critic-model.md)

---

## 核心定位

仓库交付的是 **critic 侧**：把 VLA RL 里原本的 MLP / 单帧 critic 换成历史感知的 world critic。**VLA 主干（π₀ / π₀.₅ / OpenVLA-OFT）与其 RL 训练栈仍需自备**，仓库负责 critic 的训练、评测与诊断可视化。

---

## 仓库入口（README）

| 步骤 | 脚本 | 说明 |
|------|------|------|
| 1 | `1_add_returns.sh` | 数据预处理：给轨迹打折扣回报标签 |
| 2 | `2_run_train.sh` | 训练启动器（1 GPU 或 8 GPU） |
| 3 | `3_run_eval.sh` | 对指定 checkpoint 评测 |
| 4 | `4_gen_video.sh` | 生成 episode 价值曲线视频（成功 / 失败对照） |

目录：`world_critic/`（核心实现）、`configs/`（训练配置 YAML）、`scripts/`（工具）、`assets/`（图与标签文件）、`episode_value_video/`（可视化模块）。

---

## 开源边界（2026-08-04 核查）

- **代码：** 完整，无占位。
- **权重 / 数据：** HF collection 已放 pick-and-place 与 LIBERO-Plus 相关资产；论文写明其余 checkpoint **逐步开源**。
- **因此：** 归类为 **部分开源**。想完整复现表 1–3 需要自己接 VLA 主干与对应 RL 算法（Flow-SDE / PPO / AWR / RECAP）。

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-wcm-world-critic-model](../../wiki/entities/paper-wcm-world-critic-model.md) | 论文实体与结论 |
| [openvla](../../wiki/entities/openvla.md) | 自回归主干（论文用 OpenVLA-OFT） |
| [paper-pi05-open-world-vla](../../wiki/entities/paper-pi05-open-world-vla.md) | flow matching 主干（π₀.₅） |
| [model-based-rl](../../wiki/methods/model-based-rl.md) | 对照：WCM 的世界模型只做 critic 表征监督，不做规划 rollout |
