# unitreerobotics（Unitree Robotics 官方 GitHub 组织）

> 来源归档

- **标题：** Unitree Robotics（unitreerobotics）
- **类型：** repo（GitHub **组织**总览，非单一仓库）
- **机构：** 宇树科技（Unitree）
- **链接：** <https://github.com/unitreerobotics>
- **官网：** <https://www.unitree.com>
- **开发者文档：** <https://support.unitree.com/home/zh/developer>
- **Hugging Face：** <https://huggingface.co/unitreerobotics>
- **公开仓库数：** 约 **52**（截至 2026-07-24）
- **wiki 策略：** 主线仓升格为**有详情的独立节点**；产品线多仓（Z1 / 灵巧手 / UniLidar / SDK2 C++·Python）**合并**叙述；周边与过时仓**仅本目录归档**，不建 stub。
- **入库日期：** 2026-04-11；深度补全 2026-07-20；全仓归档 + 去重深化 2026-07-24
- **沉淀到 wiki：** 是 → [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)
- **应用商店：** [unitree-unistore](../sites/unitree-unistore.md)

---

## 开源状态（2026-07-24）

- 绝大多数研发仓公开；UnifoLM 权重/数据在 Hugging Face。
- `unitree_model` GitHub **deprecated** → HF `unitreerobotics/unitree_model`。
- 成品技能分发见 UniStore。

---

## 已升格详情节点（wiki）

见 [`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)「wiki 独立节点（有详情）与归档策略」。

关键节点：

| 主题 | wiki |
|------|------|
| SDK2（含 Python） | [unitree-sdk2.md](../../wiki/entities/unitree-sdk2.md) |
| ROS2 / ROS1 | [unitree-ros2.md](../../wiki/entities/unitree-ros2.md) / [unitree-ros.md](../../wiki/entities/unitree-ros.md) |
| MuJoCo Sim2Sim | [unitree-mujoco.md](../../wiki/entities/unitree-mujoco.md) |
| RL 三线 | [unitree-rl-gym.md](../../wiki/entities/unitree-rl-gym.md) · [unitree-rl-lab.md](../../wiki/entities/unitree-rl-lab.md) · [unitree-rl-mjlab.md](../../wiki/entities/unitree-rl-mjlab.md) |
| 遥操作 / IL | [xr-teleoperate.md](../../wiki/entities/xr-teleoperate.md) · [unitree-sim-isaaclab.md](../../wiki/entities/unitree-sim-isaaclab.md) · [unitree-lerobot.md](../../wiki/entities/unitree-lerobot.md) |
| UnifoLM | [unifolm-vla.md](../../wiki/entities/unifolm-vla.md) · [unifolm-world-model-action.md](../../wiki/entities/unifolm-world-model-action.md) |
| 感知 / 臂 / 手 | [unilidar-sdk2.md](../../wiki/entities/unilidar-sdk2.md) · [point-lio-unilidar.md](../../wiki/entities/point-lio-unilidar.md) · [z1-sdk.md](../../wiki/entities/z1-sdk.md) · [unitree-dexterous-hand-services.md](../../wiki/entities/unitree-dexterous-hand-services.md) |

各单仓 `sources/repos/<name>.md` 的「沉淀到 wiki」字段指向上表合并页或标注仅归档。
