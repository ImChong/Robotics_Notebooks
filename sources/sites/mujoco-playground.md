# MuJoCo Playground 官方项目页

> 来源归档；本次仅完成项目页及官方仓库 README 核查，论文全文与源码运行流程尚待核验。

- **标题：** MuJoCo Playground
- **类型：** 项目主页
- **链接：** https://playground.mujoco.org/
- **论文：** https://arxiv.org/abs/2502.08844
- **技术报告：** https://playground.mujoco.org/assets/playground_technical_report.pdf
- **代码：** https://github.com/google-deepmind/mujoco_playground （公开，Apache-2.0；地形纹理另按仓库说明使用 CC0）
- **入库日期：** 2026-11-01
- **已有实体：** [MuJoCo Playground](../../wiki/entities/mujoco-playground.md)
- **代码归档：** [mujoco_playground](../repos/mujoco_playground.md)

## 项目页与官方仓库核查

- 官方项目页将其介绍为基于 MJX 的开源机器人学习框架，涵盖四足、人形、灵巧手及机械臂，目标是连接仿真、训练和 sim-to-real。
- 项目页链接代码、论文以及 locomotion、manipulation、vision 等教程，展示状态与视觉输入的零样本迁移示例；这些示例不代表所有任务都能直接迁移。
- 截至 2026-11-01，官方 README 说明训练支持 MJX JAX 和 MuJoCo Warp，提供 `train-jax-ppo --env_name CartpoleBalance` 与 `--impl warp` 示例；还提供 `train-rsl-ppo` 示例。
- README 提醒 Ampere 架构 GPU 的 JAX 默认 TF32 矩阵乘法可能影响复现性，可按其建议设置 `JAX_DEFAULT_MATMUL_PRECISION=highest`；论文复现须留意其对训练脚本差异的说明。

## 待完成

- 论文全文（含实验和局限）未能获取和通读；技术报告 PDF 获取时被服务端大小限制阻断。
- 未逐文件阅读官方训练源码，不能确认运行时调用链；待核实后更新已有实体页的项目流程图和源码运行时序图。
- 尚未执行仓库要求的 `make ci-preflight`；本环境无法完成仓库克隆。
