# AMP_mjlab 训练流程讲解视频（ManimGL）

基于 [ccrpRepo/AMP_mjlab](https://github.com/ccrpRepo/AMP_mjlab) 实际源码，用 [3b1b/manim](https://github.com/3b1b/manim)（ManimGL）生成训练管线讲解动画。

## 依赖

- Python 3.12+
- `pip install manimgl`（包名 `manimgl`，非 Manim Community）
- 系统：`ffmpeg`、`libpango1.0-dev`、`python3-dev`

## 渲染

```bash
cd media/manim/amp-mjlab
manimgl amp_mjlab_training_flow.py AmpMjlabTrainingFlow -w -m   # 1080p
# manimgl amp_mjlab_training_flow.py AmpMjlabTrainingFlow -w -l # 低清预览
```

输出默认写入 `videos/AmpMjlabTrainingFlow.mp4`；发布副本在 [`../../videos/amp-mjlab/amp-mjlab-training-flow.mp4`](../../videos/amp-mjlab/amp-mjlab-training-flow.mp4)。

## 场景内容（对照源码）

| 章节 | 对应代码 |
|------|----------|
| 训练入口 | `scripts/train.py` → `launch_training` → `run_train` |
| 任务注册 | `src/tasks/amp_loco/config/g1/__init__.py` |
| 并行环境 | `env_cfgs.py` + `amp_env_cfg.py`（4096 env、50 Hz、WalkandRun/Recovery） |
| 网络与观测 | `rl_cfg.py`（Actor 384→29、Discriminator、AMPPPO） |
| 训练迭代 | `rsl_rl/runners/amp_on_policy_runner.py` `learn()` |
| 奖励与 AMP | `amp_env_cfg.py` rewards + `predict_amp_reward` |
| Recovery Jump | README / TensorBoard `Train/mean_reward` |
| 导出部署 | `src/tasks/amp_loco/rl/runner.py` `save()` → ONNX |
