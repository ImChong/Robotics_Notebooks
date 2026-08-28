# pollen-robotics/microduck_rl

> 来源归档

- **标题：** Microduck RL
- **类型：** repo
- **组织：** pollen-robotics
- **链接：** https://github.com/pollen-robotics/microduck_rl
- **项目页：** https://pollen-robotics.com/microduck
- **Runtime：** https://github.com/pollen-robotics/microduck
- **许可：** Apache-2.0（软件）；README 另写硬件设计文件 **CC BY-SA-NC**
- **语言：** Python
- **默认分支：** `develop`（训练入口与 `AGENTS.md` 以该分支为准）
- **Stars / Forks：** ~307 / 38（2026-08-28，GitHub API）
- **入库日期：** 2026-08-28
- **一句话说明：** Microduck 的 mjlab（MuJoCo Warp）+ PPO 训练仓：50 Hz 出策略、ONNX 导出给 Runtime；菜谱核心是 BAM XL330、域随机化、齿轮背隙与一套从真机失败里炼出来的奖励约定。
- **开源状态：** **已开源**（可运行：`uv run train / play / scripts/export.py / infer_policy.py`；无 GPU 可 `--hf-jobs`）
- **项目页归档：** [pollen-robotics-microduck.md](../sites/pollen-robotics-microduck.md)
- **沉淀到 wiki：** [pollen-microduck-rl](../../wiki/entities/pollen-microduck-rl.md)

---

## 定位

为 [Microduck](https://github.com/pollen-robotics/microduck)（约 800 g、25 cm 双足）提供 RL 环境。依赖 [mjlab](https://github.com/mujocolab/mjlab) 与 [Rhoban BAM](https://github.com/Rhoban/bam)。官方 playbook 在仓内 **`AGENTS.md`**（`CLAUDE.md` 在 `develop` 上是短指针）。

## 可运行入口

需要 CUDA GPU（MuJoCo Warp）；包管理用 [uv](https://docs.astral.sh/uv/)。

```bash
uv run train Mjlab-Velocity-Flat-MicroDuck --env.scene.num-envs 4096
uv run play Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <entity/project/run_id>
uv run scripts/export.py Mjlab-Velocity-Flat-MicroDuck --wandb-run-path <...>
uv run scripts/infer_policy.py --walking output.onnx
```

无本地 GPU：任意 `train` 加 `--hf-jobs`（见 `scripts/hf/README.md`）。长跑前先 `64 envs × 5 iter` smoke。

## 任务族（`uv run list-envs` 为活注册表）

主任务在 id 中插入 `-Backlash-` 即得到 ±1° 齿轮间隙孪生（观测/动作维不变，ONNX 与 Runtime 无需改）。

| Task id 模式 | 说明 |
|--------------|------|
| `Mjlab-Velocity-{Flat,Rough}-MicroDuck` | 主任务：速度指令 + 头姿指令行走 |
| `Mjlab-VelStand-{Flat,Rough}-MicroDuck` | 走 + 跌倒恢复同一策略 |
| `Mjlab-StandUp-{Flat,Rough}-MicroDuck` | 俯卧/仰卧/坐起身，再站立与体姿 |
| `Mjlab-SitStand-{Flat,Rough}-MicroDuck` | 指令坐下 ↔ 站起 |
| `Mjlab-GroundPick-{Flat,Rough}-MicroDuck` | 蹲下喙尖触地再站回 |
| `Mjlab-BallKick-Flat-MicroDuck` | 踢 70 mm / 15 g 球（策略看不见球） |
| `Mjlab-Roulade-Flat-MicroDuck` | 前滚翻回脚 |
| `Mjlab-Velocity-Flat-MicroDuck-Rollers` 等 | 轮滑速度、swizzle、下蹲滑、下坡、轮上起身、原地转 |

部署侧用 **共享 61 维 actor 观测** 热切换 walk / recover / trick。`infer_policy.py` 在 CPU MuJoCo 里排练同一合同。

## 执行器与模型

- 全部任务用 BAM **M6** 电压控制律建模 **Dynamixel XL330**（反电动势、Coulomb/Stribeck/负载摩擦），再叠电池电压、负载压降、指令延迟、摩擦幅值 DR（`FrictionDRBamActuator`）。
- MJCF 从 Onshape 经 [onshape-to-robot](https://github.com/Rhoban/onshape-to-robot) 导出：`robot_walk.xml`（行走，躯干/头接触精简）、`robot_allcollisions.xml`（可躺地）、`robot_*_rollers.xml`、`add_backlash.py` 生成背隙变体。
- 背隙：每路伺服串联无驱 `passive_<joint>_backlash` 铰；真机编码器在间隙输出侧，观测读 `qpos[servo]+qpos[backlash]`。

## 工程不变量（摘自 AGENTS.md）

- 观测布局全家族共享：48 本体感觉 + `[twist(3), head_pose(4), body_pose(6)]`；不用的指令槽 **零填充**，禁止删维。
- 关节顺序（walk 模型）：0–4 左腿，5–8 颈/头，9–13 右腿；有 `passive_*` 时禁止写死下标。
- 必须走 `scripts/export.py` 把观测归一化烤进 ONNX；viewer `play` 会掩盖手转 checkpoint 的 bug。
- 训练默认 **不对动作低通**；训练/部署滤波不一致会直接断 sim2real。
- BAM 下 `dof_frictionloss` 被置零，摩擦 DR 必须打 `friction_scale`。

## 对 wiki 的映射

| 主题 | wiki |
|------|------|
| 训练栈与奖励课 | `wiki/entities/pollen-microduck-rl.md` |
| 整机 Runtime | `wiki/entities/pollen-microduck.md` |
| 框架 | `wiki/entities/mjlab.md` |
| 执行器模型 | `wiki/entities/bam-better-actuator-models.md` |
| 奖励 / gap | `wiki/concepts/reward-design.md`、`wiki/queries/sim2real-gap-reduction.md` |
