# ai_sapiens（ROBOTIS AI Sapiens K1 ROS 2）

> 来源归档

- **标题：** Open Humanoid AI Sapiens
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/ai_sapiens
- **机构：** 乐百机器人（ROBOTIS）
- **Stars：** ~21（2026-08）
- **许可：** Apache-2.0
- **产品页：** https://www.robotis.com/en/product/ecosystem-aisapiens.php
- **文档：** https://docs.robotis.com/docs/systems/aisapiens/introduction
- **入库日期：** 2026-08-07
- **一句话说明：** ROBOTIS AI Sapiens K1 官方 ROS 2 包：描述、bringup（`k1.launch.py`）、关节组阻抗控制器与 RC broadcaster；对接 Physical AI Tools / MuJoCo menagerie。
- **沉淀到 wiki：** [robotis-ai-sapiens](../../wiki/entities/robotis-ai-sapiens.md)

---

## 核心定位

人形 **AI Sapiens** 开源软件入口（相对 AI Worker 半人形线）：

- `ai_sapiens_bringup` — `config/k1_rev1`、`launch/k1.launch.py`
- `ai_sapiens_controllers` — `ai_sapiens_joint_group_impedance_controller`、`ai_sapiens_rc_broadcaster`
- 元包 `ai_sapiens`、CI `ai_sapiens_ci.repos`

`soma-retargeter` 子模块引用本仓 URDF/STL 作为重定向目标之一。

---

## 开源状态

**已开源** — Apache-2.0；产品文档与 GitHub 链齐全。仓库较新，任务/策略示例仍以配套文档与其它 Cyclo 仓为准。

---

## 对 wiki 的映射

- **wiki/entities/robotis-ai-sapiens.md**（新建）
- **wiki/entities/robotis.md** / **robotis-ai-worker.md** — 产品线对照
- **sources/repos/soma_retargeter.md** — 重定向消费本仓资产
