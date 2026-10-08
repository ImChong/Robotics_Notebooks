# rai-opensource/sumo：全身移动操作研究代码

- **类型：** repo
- **URL：** <https://github.com/rai-opensource/sumo>
- **项目页：** <https://sumo.rai-inst.com/>
- **论文：** <https://arxiv.org/abs/2604.08508>
- **核查日期：** 2026-10-08
- **Wiki：** [Sumo](../../wiki/methods/sumo.md)
- **配套来源：** [论文归档](../papers/sumo.md)、[RAI 路线核查](../sites/rai-institute.md)

## README 核查

官方项目页的 Code 按钮指向此仓；README 提供 `pixi install`、`pixi run build` 和 `pixi run sumo`，仿真 GUI 可选 `task=spot_box_push` 或 `task=g1_box optimizer=mppi`。构建需要本地 G1 extension 和 judo 的 MuJoCo extension。

Headless 入口为 `pixi run python -m sumo.run_mpc`，支持 `--init-task=g1_door --init-optimizer=cem --num-episodes=10`；结果默认保存 HDF5，可用 `--record-all-data` 记录 rollout。

**开放边界：** 已有可运行仿真 / 采样规划研究代码；本次核查 README，没有安装和运行重型仿真环境。不以 README 仿真命令证明真实 Spot / G1 驱动、全部训练数据或论文指标均可直接复现。
