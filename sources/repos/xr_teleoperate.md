# xr_teleoperate

> 来源归档

- **标题：** xr_teleoperate
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/xr_teleoperate
- **星标（截至 2026-07-24）：** ~1572
- **最近推送：** 2026-05-28
- **主要语言：** Python
- **分类：** 遥操作与采数
- **入库日期：** 2026-07-24
- **核对范围：** 2026-11-21 通读 `unitreerobotics/xr_teleoperate@7e5026e` 官方 `README.md` 全文（含 v1.6 支持 H2/R1/BrainCo、设备支持矩阵、`teleimager v2.0.8` 钉扎与 `televuer` 证书配置、仿真与真机启动参数表、代码目录总览）及主入口 `teleop/teleop_hand_and_arm.py`（状态机、DDS domain 0/1 切换、`--ipc` 与 `--record` 控制参数）；`teleimager` / `televuer` 子模块内部与真机未运行测试。
- **通读：** 已完成
- **一句话说明：** XR（AVP/PICO/Quest）全身遥操作 G1/H1/H2/R1 的官方主仓，含仿真支持。
- **沉淀到 wiki：** 是 → [`wiki/entities/xr-teleoperate.md`](../../wiki/entities/xr-teleoperate.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)

---

## README 要点（编译自上游，@7e5026e）

- **v1.6（2026-07-29）：** 新增 H2 与 R1（`R1_A5` / `R1_A7`）支持、BrainCo 手部控制输入、默认头偏航相对臂参考。
- **支持机型与末端：** G1（29/23 DoF）、H1（4-DoF 臂）、H1_2/H2（7-DoF 臂）、R1（5/7-DoF 臂）；末端含 Dex1-1（含 `dex1_internal` 电机 31/33 内线模式）、Dex3-1、Inspire（DFX/FTP）、BrainCo。
- **依赖与通信：** 依赖 `teleimager`（钉扎 `v2.0.8`，不可升至 2.1.x）、`televuer`（Vuer + HTTPS/WebRTC）、`unitree_sdk2_python`；`--sim` 切 DDS domain 1 对接 `unitree_sim_isaaclab`，真机为 domain 0。

## 开源状态

- **已开源**：公开 GitHub 仓库（unitreerobotics/xr_teleoperate）。


## 对 wiki 的映射

- 实体页：[`wiki/entities/xr-teleoperate.md`](../../wiki/entities/xr-teleoperate.md)
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)

## 时间核对（2026-10-06）

- **代码历史起点：** 2024-08-06（UTC），官方仓库提交 [`b990c0e`](https://github.com/unitreerobotics/xr_teleoperate/commit/b990c0eff38404755a1b57bce8453a499d358315) 已含 `teleop / scripts / act` 实现。
- **时间口径：** 当前可核实的早期代码提交；不等同于正式首发、仓库首次转为公开或当前版本发布日期。仓库创建日、最近推送日与 wiki 入库日不作为首发依据。
