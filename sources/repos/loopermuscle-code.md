# LooperMuscle/Code（官方仓库）

- **标题：** LooperMuscle — Fast and Stable Learning of Humanoid Whole-Body Tracking
- **类型：** repo / humanoid / whole-body-tracking / reinforcement-learning / mixture-of-experts
- **仓库：** <https://github.com/LooperMuscle/Code>
- **论文：** <https://arxiv.org/abs/2608.00820>
- **项目页：** <https://loopermuscle.github.io/>
- **收录日期：** 2026-08-22
- **许可证：** Apache-2.0
- **维护方：** LooperMuscle 团队（DeepMirror × HKUST × MBZUAI）

## 一句话摘要

LooperMuscle 官方仓：以 **Holosoma** 为骨干提供 **WBT 训练（PPO/FastSAC）**、**MuJoCo/真机推理部署** 与 **运动重定向**；README 强调论文 MJLab 基准用特权观测，真机策略在可部署 154-D 接口上重训。

## 发布状态（2026-08-22 对照 README）

| 组件 | 状态 |
|------|------|
| `src/holosoma/` 训练框架 | ✅ `train_agent.py`；IsaacGym/IsaacSim/MJWarp |
| `src/holosoma_inference/` 推理与真机 WBT | ✅ G1 WBT + Locomotion |
| `src/holosoma_retargeting/` 重定向 | ✅ |
| `scripts/setup_inference.sh` 等 | ✅ |
| `demo_scripts/demo_lafan_wb_tracking.sh` | ✅ LAFAN WBT 演示 |
| 论文 MJLab 特权基准数字 | ⚠️ 与 Holosoma 可部署接口分离；不直接迁 checkpoint |
| BibTeX | ⏳ README：公开后补充 |

## 环境与入口

```bash
git clone https://github.com/LooperMuscle/Code.git
cd Code
bash scripts/setup_inference.sh
source scripts/source_inference_setup.sh
```

训练（Holosoma，示例 FastSAC locomotion）：

```bash
source scripts/source_isaacgym_setup.sh
python src/holosoma/holosoma/train_agent.py \
    exp:g1-29dof-fast-sac \
    simulator:isaacgym \
    logger:wandb \
    --training.seed 1
```

WBT 与真机流程见 `src/holosoma_inference/docs/workflows/`。

## 为何值得保留

- **可跑部署栈：** G1 真机 WBT 经 Holosoma + ONNX 50 Hz，与论文 Fig. 7 对应。
- **FastSAC 配方延伸：** 在 Holosoma 上与 PPO 并列，LooperMuscle 架构落在此框架内。
- **与论文口径对齐：** README 明确 MJLab vs Holosoma 观测差异，避免误把特权基准当部署预期。

## 关联资料

- 论文：[`sources/papers/loopermuscle_arxiv_2608_00820.md`](../papers/loopermuscle_arxiv_2608_00820.md)
- 项目页：[`sources/sites/loopermuscle-github-io.md`](../sites/loopermuscle-github-io.md)
- 上游运行时：[Holosoma](https://github.com/amazon-far/holosoma)
