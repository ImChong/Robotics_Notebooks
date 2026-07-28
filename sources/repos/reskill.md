# krishanrana/reskill（ReSkill 官方 PyTorch 实现）

> 来源归档

- **标题：** Residual Skill Policies (ReSkill)
- **类型：** repo
- **来源：** QUT Centre for Robotics（Krishan Rana）
- **链接：** <https://github.com/krishanrana/reskill>
- **项目页：** <https://krishanrana.github.io/reskill/> — 归档见 [`sources/sites/reskill-github-io.md`](../sites/reskill-github-io.md)
- **License：** MIT
- **入库日期：** 2026-07-28
- **一句话说明：** ReSkill（CoRL 2022）官方 PyTorch 实现：脚本控制器采数据 → VAE 技能嵌入 + real NVP 技能先验 → 高层 RL 选技能、低层残差策略细修的完整训练管线。
- **沉淀到 wiki：** [`wiki/entities/paper-reskill-residual-skill-policies.md`](../../wiki/entities/paper-reskill-residual-skill-policies.md)

---

## 核心定位

官方 PyTorch 实现（QUT Centre for Robotics Open Source 徽章）。环境沿用 Fetch 臂 MuJoCo 任务族（改编自 Silver et al. RPL 环境）：

| 阶段 | 命令 | 说明 |
|------|------|------|
| 数据采集 | `python data/collect_demos.py --num_trajectories 40000 --subseq_len 10 --task block` | 脚本控制器轨迹；`block`（用于 Stack/CleanUp/SlipperyPush）与 `hook`（用于 ComplexHook）两套；也可下载预采数据集（Google Drive） |
| 技能模块训练 | `python train_skill_modules.py --config_file block/config.yaml --dataset_name fetch_block_40000` | VAE 嵌入 + real NVP 技能先验联合训练 |
| 下游 RL | 见 README 后续命令 | 高层策略 + 低层残差策略 on-policy 训练 |

## 运行要点（README）

- **依赖：** Python 3.7+、MuJoCo 2.1（mujoco-py）、Ubuntu 18.04；`conda env create -f environment.yml && pip install -e .`。
- **环境：** `FetchPyramidStack-v0`、`FetchCleanUp-v0`、`FetchSlipperyPush-v0`、`FetchComplexHook-v0`（下游）；`FetchHook-v0`、`FetchPlaceMultiGoal-v0`（采数据）。

## 与仓库内实体的关系

| 关联 | 说明 |
|------|------|
| [paper-reskill-residual-skill-policies](../../wiki/entities/paper-reskill-residual-skill-policies.md) | 本仓库对应的论文实体页 |
| [paper-residual-policy-learning](../../wiki/entities/paper-residual-policy-learning.md) | 下游任务环境直接改编自 RPL 官方环境（k-r-allen/residual-policy-learning） |
