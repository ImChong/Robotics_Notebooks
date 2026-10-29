# RP1_description

> 来源归档（RP1 官方仓库体系 · 本体模型）

- **标题：** RP1_description — RP1 URDF / MJCF 机器人模型
- **类型：** repo（机器人描述 / 仿真资产）
- **机构：** 萝卜派对（RoboParty）
- **链接：** https://github.com/Roboparty/RP1_description （同一仓库亦以 `rp1_description` 为子模块 URL 被 [`robolab`](https://github.com/Roboparty/robolab) 引用）
- **入库日期：** 2026-10-27（核查快照：`main` 的 `d8c6dd0`，2026-10-26，提交说明 `update actuator params for new motors`）
- **一句话说明：** RP1 的 URDF、MJCF、STL 网格与简易地形资产，提供 24 DoF（`rp1_24dof`）、28 DoF（`rp1`）及带灵巧手（`*_hand`）三类模型，是 `roboparty_train` 与仿真验证共享的本体模型基线。
- **沉淀到 wiki：** 是（[`wiki/entities/rp1-description.md`](../../wiki/entities/rp1-description.md)）
- **上级入口：** [RP1 总览仓库 README](https://github.com/Roboparty/RP1)「Software & Systems」表；wiki 总览 [roboparty-rp1](../../wiki/entities/roboparty-rp1.md)

## 开源状态

- **已开源**：模型源文件（URDF / MJCF / STL / 地形 PNG）全部直接入库，未使用 Git LFS（`.gitattributes` 仅设置换行与二进制属性）。
- **许可证未在仓库内声明**：快照中无 `LICENSE`、无 `README`；复用前须向 RoboParty 确认条款（勿套用同组织其他仓库的 GPL-3.0）。
- 仓库**不含**动力学辨识数据、执行器台架曲线或装配文档。

## 关键文件清单（快照 `d8c6dd0`）

| 路径 | 内容 |
|------|------|
| `urdf/rp1_24dof.urdf` | 24 个 revolute 关节、25 个 link（含 `base_link`）；link 质量求和约 39.11 kg（本页计算，非官方整机重量） |
| `urdf/rp1.urdf` | 28 个 revolute 关节（每臂腕部 `wrist_roll` / `wrist_yaw` / `wrist_pitch` 三轴） |
| `urdf/rp1_hand.urdf` / `rp1_hand_fixed.urdf` | 含灵巧手：前者 50 个可动关节 + 12 个固定；后者 28 可动 + 34 固定（手指固定） |
| `mjcf/robot_24dof.xml`、`robot.xml`、`robot_hand.xml`、`robot_hand_fixed.xml` | 与 URDF 对应的 MuJoCo 模型 |
| `mjcf/rp1_flat.xml`、`mjcf/rp1_rough.xml` | 场景文件：平地 / 高度场地形（`terrain_assets/terrain_hfield.png`），二者均 `include` `robot.xml`（28 DoF） |
| `meshes/*.STL`、`meshes/hand/{left,right}/` | 二进制 STL（文件头 `RP1 refined binary STL`）；每侧手 16 个手指 / 掌 link |

## 对 wiki 的映射

- [wiki/entities/rp1-description.md](../../wiki/entities/rp1-description.md)
- 区分：[atom01-description](../../wiki/entities/atom01-description.md)（对应 `atom01_description`，与 `rpo_description` 提交 `8e0f8f7` 相同，属 RPO 本体）
- 下游：[roboparty_train](roboparty_train.md)（`robolab` 子模块 `data/robots/roboparty/rp1` 指向本仓）、[roboparty_deploy](roboparty_deploy.md)（关节顺序 / 增益需与模型一致）
