# YAHMP（Yet Another Humanoid Motion tracking Policy）

> 来源归档

- **标题：** YAHMP
- **类型：** repo
- **来源：** Inria（Fabio Amadio / Enrico Mingo Hoffman）
- **链接：** <https://github.com/fabio-amadio/yahmp>
- **论文 URL 别名：** 论文写 <https://github.com/hucebot/yahmp>（org fork，parent 为 `fabio-amadio/yahmp`）
- **论文：** <https://arxiv.org/abs/2607.19903>
- **许可：** Apache-2.0
- **入库日期：** 2026-07-24
- **一句话说明：** 基于 **mjlab** 的 Unitree G1 全身 general motion tracking 开源框架：模块化 MDP、PPO / Teacher–Student、ONNX 导出与 MuJoCo 部署评测，并内置与 TWIST2 ONNX 的对照脚本。
- **沉淀到 wiki：** [`wiki/entities/paper-yahmp.md`](../../wiki/entities/paper-yahmp.md)

---

## 核心定位

配合论文 *What Matters in Humanoid General Motion Tracking?*，把「命令 / 历史 / 动作 / 驱动 / 手部力 / 训练范式」做成可切换配置，方便在同一协议下做消融与零样本真机部署。

---

## 仓库入口（README / `src/yahmp/scripts`）

| 组件 | 说明 |
|------|------|
| 安装 | `uv sync`；`uv run list_envs \| rg YAHMP` 校验环境 |
| 运动资产 | Google Drive 包 → `assets/motions/g1_omomo_amass_clean/` |
| 训练 | `uv run train Mjlab-YAHMP-Unitree-G1 --env.scene.num-envs 8192` |
| ONNX 导出 | `yahmp.scripts.deploy.export_checkpoint_to_onnx`（本地 ckpt 或 W&B run） |
| MuJoCo 部署 | `yahmp.scripts.deploy.run_yahmp_onnx_mujoco`；预置 `assets/models/g1_yahmp.onnx` |
| TWIST2 对照 | `run_twist2_onnx_mujoco` + `evaluate_twist2_onnx_success_parallel` |
| 变体 | `YAHMP-Future`；Teacher / Student（Action-Matching / KL-Matching）环境 ID |

---

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-yahmp](../../wiki/entities/paper-yahmp.md) | 论文实体与消融结论 |
| [mjlab](../../wiki/entities/mjlab.md) | 底层 GPU RL / manager-based 环境栈 |
| [paper-twist2](../../wiki/entities/paper-twist2.md) | 外部完整管线基线；仓库内带 ONNX 对照评测 |
| [paper-extreme-rgmt](../../wiki/entities/paper-extreme-rgmt.md) | 同台 G1 高动态 generalist 跟踪对照（未开源） |
