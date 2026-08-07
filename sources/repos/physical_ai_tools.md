# physical_ai_tools（ROBOTIS Physical AI Tools）

> 来源归档

- **标题：** ROBOTIS Physical AI Tools
- **类型：** repo
- **链接：** https://github.com/ROBOTIS-GIT/physical_ai_tools
- **默认分支（文档）：** `jazzy`（README：`git clone -b jazzy ... --recursive`）
- **机构：** 乐百机器人（ROBOTIS）
- **Stars：** ~140（2026-08）
- **许可：** Apache-2.0
- **主页：** https://ai.robotis.com/
- **入库日期：** 2026-08-07
- **一句话说明：** LeRobot + ROS 2 的 Physical AI 开发界面：含 `physical_ai_server`、行为树包 `physical_ai_bt`、lerobot 子模块与 Docker/s6。
- **沉淀到 wiki：** [robotis-physical-ai-tools](../../wiki/entities/robotis-physical-ai-tools.md)

---

## 核心定位

面向 AI Worker / Open Manipulator 等平台的 **数据采集—训练—推理界面层**（相对 `cyclo_intelligence` 更偏「工具 + BT 宏动作」而非完整 Cyclo Brain 容器矩阵）。

可见顶层：

| 路径 | 角色 |
|------|------|
| `lerobot/` | LeRobot 子模块 |
| `physical_ai_bt/` | BT 节点：`move_arms` / `move_head` / `move_lift` / `rotate` 等 |
| `docker/` | Compose + s6 `physical_ai_server` |
| 文档 | [ai.robotis.com](https://ai.robotis.com/) |

与 [ai_worker](https://github.com/ROBOTIS-GIT/ai_worker)、[cyclo_lab](https://github.com/ROBOTIS-GIT/cyclo_lab) Sim2Real 说明交叉引用。

---

## 开源状态

**已开源** — Apache-2.0；含可辨识 ROS 2 / Docker 入口。

---

## 对 wiki 的映射

- **wiki/entities/robotis-physical-ai-tools.md**（新建）
- **wiki/entities/cyclo-intelligence.md** — 更完整 BT+VLA 栈对照
- **wiki/entities/lerobot.md** — 后端
