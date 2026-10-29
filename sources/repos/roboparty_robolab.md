# robolab（Roboparty：AMP / BeyondMimic / Parkour 训练任务仓）

> 来源归档（RP1 总览 README「Demo」表的三个子项目共用此仓库）

- **标题：** robolab — RoboParty Isaac Lab 任务库（RPO / RP1 人形）
- **类型：** repo
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/robolab
- **RP1 README 引用的固定提交：** `6b1c3d9988497c8961dcba77892de32edc1770e1`（2026-07-26，`fix bugs`）下的 `robolab/tasks/manager_based/{amp,beyondmimic,parkour}`
- **入库日期：** 2026-10-27（核查 HEAD `fcf925c`，2026-10-26，`bump RP1 description submodule`）
- **许可：** `LICENSE` 为 BSD-3-Clause（版权行含 Isaac Lab Project Developers 与 RoboLab Project Developers）
- **一句话说明：** 基于 Isaac Lab 的人形训练任务库，`manager_based/` 下分 AMP、BeyondMimic、Parkour 三套任务，另有 `direct/`（基础 locomotion 与 interrupt）；机器人为 RPO（23 关节）与 RP1 24DOF。
- **沉淀到 wiki：**
  - [`wiki/entities/roboparty-robolab-amp.md`](../../wiki/entities/roboparty-robolab-amp.md)
  - [`wiki/entities/roboparty-robolab-beyondmimic.md`](../../wiki/entities/roboparty-robolab-beyondmimic.md)
  - [`wiki/entities/roboparty-robolab-parkour.md`](../../wiki/entities/roboparty-robolab-parkour.md)

> 与 [NVLabs RoboLab（VLA 评测基准）](../../wiki/entities/robolab.md) 同名但无关。

---

## 开源状态核查（2026-10-27）

- **已开源**：任务配置、奖励/观测/终止项、训练与回放入口、动作数据转换脚本均在仓库内。
- **仓库无 README**（根目录仅 `LICENSE`、`setup.py`、`data/`、`robolab/`、`scripts/`）；入口说明来自 RP1 总览 README 与代码本身。`rp1_amp_env_cfg.py` 注释提到 “See the data README”，但仓库中未见该文件。
- **机器人资产为未初始化的子模块**：`.gitmodules` 指向 `Roboparty/rpo_description` 与 `Roboparty/rp1_description`（挂载到 `data/robots/roboparty/{rpo,rp1}`）；浅克隆下目录为空，训练前需自行拉取子模块。
- **动作数据随仓库提供**：`data/motions/` 下 `rp1_amp`（8 个 pkl）、`rp1_bm`（2 个 npz）、`rpo_amp`（30 个 pkl）、`rpo_bm`（getup 的 csv/npz 与 yundong0/1 npz）。
- **安装要求**：`setup.py` 声明 `python_requires>=3.12,<3.13`，入口点 `isaaclab.tasks: robolab = robolab.tasks`；训练/回放脚本 `scripts/rsl_rl/{train,play}.py` 调用 `isaaclab_rl.entrypoints`（Isaac Lab 3 一侧）。AMP 与 Parkour 的算法类名为 `rsl_rl.algorithms.ppo_amp:PPOAMP`，对应 RSL-RL 的 AMP 变体由外部安装，未在本仓库内（版本未核实）。

## 固定提交与 HEAD 的差异

README 链接的 `6b1c3d9` 到 HEAD 之间有 9 个提交（2026-10-24 至 2026-10-26），涉及“迁移到 Isaac Lab 3”、新增 RP1 资产与任务、HHTools 动作转换并移除旧重定向、更新 RP1 执行器包络参数。`manager_based/` 下共 43 个文件有改动（+1524/−564 行）。

| 子项目 | 固定提交 `6b1c3d9` | HEAD `fcf925c` |
|--------|-------------------|----------------|
| AMP | 仅 `RPO-AMP` | 增加 `RP1-AMP`、`RP1-AMP-Play` |
| BeyondMimic | `RPO-BeyondMimic`、`RPO-Getup-Mimic` | 增加 `RP1-BeyondMimic` |
| Parkour | `RPO-Parkour`、`RPO-Parkour-Play` | 仍仅 RPO（与 README 的 “RP1 support coming soon” 一致） |

三个子项目的 wiki 页以 HEAD 为准，并注明固定提交的差异。

## 注册的任务 ID（HEAD）

`RP1-AMP` / `RP1-AMP-Play` / `RPO-AMP` / `RPO-AMP-Play` / `RPO-BeyondMimic` / `RPO-Getup-Mimic` / `RP1-BeyondMimic` / `RPO-Parkour` / `RPO-Parkour-Play`（另有 `direct/` 下的 locomotion 与 interrupt 任务，不属于 README 三个 Demo）。

## 工具

`scripts/tools/hhtools/convert_motion.py`：把 HHTools 导出的机器人 CSV/PKL 转为 AMP `.pkl` 或 BeyondMimic `.npz`；参数 `--robot {rpo,rp1}`、`--input_file/-f`、`--output_file/-o`、`--output_fps`、`--loop {clamp,wrap}`。上游工具见 [human-humanoid-tools](../../wiki/entities/human-humanoid-tools.md)。

---

## 对 wiki 的映射

- [roboparty-robolab-amp](../../wiki/entities/roboparty-robolab-amp.md)
- [roboparty-robolab-beyondmimic](../../wiki/entities/roboparty-robolab-beyondmimic.md)
- [roboparty-robolab-parkour](../../wiki/entities/roboparty-robolab-parkour.md)
- 父级：[roboparty-rp1](../../wiki/entities/roboparty-rp1.md)、[roboparty](../../wiki/entities/roboparty.md)
- 总览仓库归档：[roboparty_rp1.md](roboparty_rp1.md)
