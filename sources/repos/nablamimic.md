# ∇Mimic 官方仓库

- **代码：** <https://github.com/OamIcs/NablaMimic>（Apache-2.0；第三方数据另有许可）
- **论文：** <https://arxiv.org/abs/2610.41654>
- **项目页：** <https://sailab.ai/nablamimic/>
- **核对日期：** 2026-11-04；源码 commit `bfbb2a6773b4776c17402a34a93ec7589d9c7c18`

README 给出 `scripts/setup_env.sh`、`scripts/download_lafan1.sh`、训练入口 `python -m nablamimic.train --config configs/ours.args --out runs/ours_s0` 和评估入口 `scripts/evaluate_run.sh`。代码已公开；默认配置约需 50 GB 显存，数据许可证需另行遵守。

`src/nablamimic/train.py` 构建动作库和 `TrackingEnv`，经 `sim.py` 创建 `nablaengine.jax_ffi.EnginePhysics`；rollout 通过 `jax.lax.scan` 展开，`jax.value_and_grad` 求导并构建动作回归目标，另行更新 critic。`sim.py` 的批量 step 将控制目标传给引擎；评估脚本调用 `nablamimic.evaluate`。见 [实体节点](../../wiki/entities/paper-nablamimic.md) 的源码时序图。未在本环境运行训练。
