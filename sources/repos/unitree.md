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
| UnifoLM | [unifolm-vla.md](../../wiki/entities/unifolm-vla.md) · [unifolm-world-model-action.md](../../wiki/entities/unifolm-world-model-action.md) · [unifolm-wla.md](../../wiki/entities/unifolm-wla.md) |
| 感知 / 臂 / 手 | [unilidar-sdk2.md](../../wiki/entities/unilidar-sdk2.md) · [point-lio-unilidar.md](../../wiki/entities/point-lio-unilidar.md) · [z1-sdk.md](../../wiki/entities/z1-sdk.md) · [unitree-dexterous-hand-services.md](../../wiki/entities/unitree-dexterous-hand-services.md) |

各单仓 `sources/repos/<name>.md` 的「沉淀到 wiki」字段指向上表合并页或标注仅归档。

## 公司路线时间证据（2026-10-06）

日期按事件记录；代码历史起点仅证明该提交已有实现，不推定正式首发。当前八节点保留原有范围，细分如下。

| 项目 | 日期 | 事件与官方依据 |
| --- | --- | --- |
| G1 | 2024-05-13 | 官网产品发布，见 [官方产品归档](../sites/unitree-g1-official.md) |
| `unitree_rl_gym` | 2023-10-11 | [实现代码提交 `25877c7`](https://github.com/unitreerobotics/unitree_rl_gym/commit/25877c7eaf92d40ddd78313b261c7a1074905899)；代码历史起点，非首发结论 |
| `xr_teleoperate` | 2024-08-06 | [实现代码提交 `b990c0e`](https://github.com/unitreerobotics/xr_teleoperate/commit/b990c0eff38404755a1b57bce8453a499d358315)；代码历史起点，非首发结论 |
| `unitree_lerobot` | 2024-10-18 | [实现代码提交 `7693322`](https://github.com/unitreerobotics/unitree_lerobot/commit/76933229e10efa3d47755555797e27f08495cd0a)；代码历史起点，非首发结论 |
| `unitree_sim_isaaclab` | 2025-06-24 | [实现代码提交 `d9a48e9`](https://github.com/unitreerobotics/unitree_sim_isaaclab/commit/d9a48e9abfebe86a13f8dc91ea989261960a04a5)；代码历史起点，非首发结论 |
| UnifoLM-WMA-0 | 2025-09-15；2025-09-22 | [README News](https://github.com/unitreerobotics/unifolm-world-model-action#news)：先开放训练/推理与权重，后开放机器人部署代码 |
| UnifoLM-VLA-0 | 2026-01-29 | [README News](https://github.com/unitreerobotics/unifolm-vla#news)：训练/推理代码与模型权重开放 |
| UnifoLM-WLA-1.0 | 2026-09-11；2026-09-20；2026-09-28 | [README News](https://github.com/unitreerobotics/unifolm-wla#-news)：ER 系列权重 → 模型模块/动作专家训练代码 → WLA-1.0-Base 权重与微调代码 |

G1 是产品发布，UnifoLM 是官方资产开放事件，四个工程仓是代码历史起点。不得把这三种日期读成统一的模型发布时间。WLA 的 2026-09-18 入库快照见 [原归档](unifolm-wla.md)，后续代码开放已补核；不沿用当时的「后训练待发布」作为现状。
