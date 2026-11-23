# unitree_lerobot

> 来源归档

- **标题：** unitree_lerobot
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/unitree_lerobot
- **星标（截至 2026-07-24）：** ~725
- **最近推送：** 2026-05-25
- **主要语言：** Python
- **分类：** 模仿学习
- **入库日期：** 2026-07-24
- **核对范围：** 2026-11-21 通读 `unitreerobotics/unitree_lerobot@41c2805` 官方 `README.md` 全文（v0.1–v0.3 版本演进、环境安装、数据编辑器、JSON→LeRobot 转换、ACT/Diffusion/Pi0/Pi05/GR00T 训练与 `eval_g1.py` / `eval_g1_sim.py` / `eval_g1_dataset.py` / `replay_robot.py` 入口）及真机评测入口 `unitree_lerobot/eval_robot/eval_g1.py`；`lerobot` 子模块（钉扎 `0878c68`）内部与真机未运行测试。
- **通读：** 已完成
- **一句话说明：** 基于 Hugging Face LeRobot 的 G1 双臂灵巧手模仿学习训练/测试官方改版。
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree-lerobot.md`](../../wiki/entities/unitree-lerobot.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)

---

## README 要点（编译自上游，@41c2805）

- **版本演进：** v0.1 支持 G1 + Dex1/Dex3；v0.2 增 BrainCo/Inspire 手、数据集回放与 `unitree_sim_isaaclab` 仿真验证；v0.3 升级 LeRobot dataset v3.0 并增 `pi05` / `groot` 策略支持。
- **核心入口：** `data_editor_EN.py` 裁剪坏轨迹、`utils/sort_and_rename_folders.py` 与 `convert_unitree_json_to_lerobot.py` 转格式、`lerobot_train.py` 训练、`eval_robot/eval_g1.py`（真机）/ `eval_g1_sim.py`（仿真）/ `eval_g1_dataset.py`（离线）/ `replay_robot.py`（回放）。

## 开源状态

- **已开源**：公开 GitHub 仓库（unitreerobotics/unitree_lerobot）。


## 对 wiki 的映射

- 实体页：[`wiki/entities/unitree-lerobot.md`](../../wiki/entities/unitree-lerobot.md)
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)

## 时间核对（2026-10-06）

- **代码历史起点：** 2024-10-18（UTC），官方仓库提交 [`7693322`](https://github.com/unitreerobotics/unitree_lerobot/commit/76933229e10efa3d47755555797e27f08495cd0a) 已含 `lerobot / unitree_utils` 实现。
- **时间口径：** 当前可核实的早期代码提交；不等同于正式首发、仓库首次转为公开或当前版本发布日期。仓库创建日、最近推送日与 wiki 入库日不作为首发依据。
