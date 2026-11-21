# UrbanVLA

> 来源归档（国内具身开源全景 · 源码核对）

- **标题：** UrbanVLA
- **类型：** repo
- **机构：** 银河通用（Galbot · 北京大学 · USTC · BAAI）
- **链接：** <https://github.com/GalaxyGeneralRobotics/UrbanVLA>
- **论文与项目页：** [arXiv:2510.23576](https://arxiv.org/abs/2510.23576)（ICRA 2026）· <https://pku-epic.github.io/UrbanVLA-Web/>
- **许可：** Apache-2.0
- **分类：** VLA/操作模型
- **入库日期：** 2026-09-06
- **核查日期：** 2026-11-21
- **一句话说明：** UrbanVLA 官方仓库当前公开 MetaUrban 数据采集、99 分位轨迹归一化后处理与 PointNav/SocialNav 评测基准流水线；UrbanVLA 模型结构、训练脚本与权重仍处于 README TODO 待发布状态。
- **核对范围：** 通读 `GalaxyGeneralRobotics/UrbanVLA@c75294a` 的 `README.md`、`evaluation.md`、`scripts/collect.py`、`benchmark/metaurban_eval.py`、`benchmark/policy/__init__.py`、`benchmark/eval.sh` 及 `data_postprocess/common/` 归一化契约；未在本机编译 C++ ORCA 扩展或运行 Panda3D/MetaUrban 仿真。
- **通读：** 已完成

## 开源状态

- **部分开源（Apache-2.0）**：
  - **已发布：** MetaUrban 环境打包与 ORCA 编译脚本、A\*/PPO 专家统一采集 CLI（`scripts/collect.py`）、99 分位归一化与 Sekai 轨迹清洗工具（`data_postprocess/`）、MetaUrban 4 组基准评测流水线（`benchmark/eval.sh`, `benchmark/metaurban_eval.py`）与 `NavigationPolicy` 插入契约（附带 `dummy` 与 `metaurban_ppo` 权重）。
  - **未发布（README TODO）：** Sekai 后处理完整说明、UrbanVLA 模型结构与 SFT/RFT 训练代码、UrbanVLA 预训练权重。

## 关键入口与实现要点（`c75294a`）

- **采集：** `python scripts/collect.py --policy {astar,rl} --env {static,dynamic} --view_mode {front,panorama}` 分发至 `astar_collect.py` 或 `rl_expert_collect.py`。
- **归一化契约：** `data_postprocess.metaurban.{astar,rl_expert}.post_process` 统计未来 8 步相对位姿 `(dx, dy, dyaw)` 的第 99 百分位 `NORM_SCALE_X/Y/THETA`，评测端 `NavigationPolicy` 子类必须显式对齐同一组常数。
- **评测：** `bash benchmark/eval.sh {pointnav-test,pointnav-unseen,socialnav-test,socialnav-unseen} <gpus>` 按 X/C/S 地图分片并行启动 `benchmark/metaurban_eval.py`，通过 `PolylineProcessor.simulate_sd_noise_pipeline` 注入仿真路书噪声，调用 `policy.predict()` 输出 8 步归一化航点并插值为 15 点执行，写出 `run_manifest.json`、`info.json` 与 `metric.json`（含 `sr`/`spl`/`rc`/`cost`/`sns`）。

## 对 wiki 的映射

- [wiki/entities/cn-os-urbanvla.md](../../wiki/entities/cn-os-urbanvla.md)
