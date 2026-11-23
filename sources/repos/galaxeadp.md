# GalaxeaDP

> 来源归档（国内具身开源全景）

- **标题：** GalaxeaDP
- **类型：** repo
- **机构：** 星海图
- **链接：** https://github.com/OpenGalaxea/GalaxeaDP
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **核对范围：** 已克隆 `OpenGalaxea/GalaxeaDP@21abf51`（核对 `README.md`、`scripts/sim_r1pro_blocks.sh`、`src/train.py`；未在本地执行仿真采集与 GPU 训练）。
- **通读：** 已完成
- **一句话说明：** 星海图（Galaxea AI）开源的 Diffusion Policy（DP）模块化训练与评估仓库，基于 Hydra + PyTorch Lightning，贯通 GalaxeaManipSim 仿真数据与 GalaxeaLeRobot 数据集，MIT 许可。
- **沉淀到 wiki：** [`wiki/entities/cn-os-galaxeadp.md`](../../wiki/entities/cn-os-galaxeadp.md)

## 开源状态

- **已开源**：公开仓库（MIT License，<https://github.com/OpenGalaxea/GalaxeaDP>）。

## 一手仓库要点摘录（`OpenGalaxea/GalaxeaDP@21abf51`）

- **训练与评估管线（`scripts/sim_r1pro_blocks.sh`）**：完整覆盖演示采集（`collect_demos`）→ 回放验证 → 转 LeRobot 格式 → `src/train.py` 训练 → 开环 HTML 可视化评估 → 仿真闭环策略评估（支持关节位置 / 末端位姿控制）。
- **真机转换状态**：README 标注面向真实机器人 MCAP 数据的转换与训练流程仍处于 `Coming soon`。

## 对 wiki 的映射

- [wiki/entities/cn-os-galaxeadp.md](../../wiki/entities/cn-os-galaxeadp.md)
