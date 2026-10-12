# LeWAM（MinghaoFu/lewam）

> 来源归档

- **标题：** LeWAM
- **类型：** repo
- **来源：** UCSD / ETH / UBC / Brown（Minghao Fu、Tavis Siebert、Eryk Halicki、Randall Balestriero）
- **链接：** <https://github.com/MinghaoFu/lewam>
- **代码：** <https://github.com/MinghaoFu/lewam>
- **论文：** <https://arxiv.org/abs/2610.12818>（见 [lewam_arxiv_2610_12818.md](../papers/lewam_arxiv_2610_12818.md)）
- **项目页：** <https://le-wam.github.io/>（见 [le-wam-github-io.md](../sites/le-wam-github-io.md)）
- **许可：** MIT（部分 encoders/module 代码来自 le-wm）
- **入库 / 核查日期：** 2026-10-12（浅克隆 tip `9d75709c`，提交日 2026-10-10）
- **沉淀到 wiki：** [`wiki/entities/paper-lewam.md`](../../wiki/entities/paper-lewam.md)

## 仓库结构（核查）

| 路径 | 作用 |
|------|------|
| `install.sh` | `bash install.sh lewam cu128` 建 conda 环境；依赖 `stable-worldmodel[format,train]==0.1.1` |
| `scripts/decompress_h5.py` | 解压并准备 HDF5 数据集（`.h5.zst`） |
| `scripts/train_lewam.py` + `configs/train/{base,gr,tc}.yaml` | 训练：`gr` 目标条件视觉规划，`tc` 无目标任务完成 |
| `lewam/models/lewam.py` | `LeWAM`：MoT 预测器、`loss`、`predict_actions`、`predict_states` |
| `lewam/models/encoders.py`、`module.py` | 编码器与模块（部分源自 le-wm） |
| `scripts/eval_lewam.py` + `configs/eval/*.yaml` | 评测：`eval.mode=lewam_policy` 或 `lewam_plan`；`plan_mode` 为 best_of_k / grad / steer / cem |
| `lewam/eval/policy.py`、`planners.py` | `LeWAMPolicy`、`LeWAMPlanner` |
| `lewam/envs/` | Push-T、Cube、Scene、Puzzle、PointMaze、Reacher、DexMimicGen、Robomimic 等环境 |
| `docs/DATA.md`、`docs/REPRODUCE.md` | 数据格式与复现设置 |

## 开放范围

- 训练、评测、环境与配置齐全；README 列出各任务检查点与数据集 HF 链接（如 lewam-tworoom、lewam-pusht）。
- 仓库为仿真基准；论文真机部署代码未在目录中见到（推测未包含）。
