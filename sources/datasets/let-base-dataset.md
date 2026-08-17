# LET-Base-Dataset（Hugging Face 数据卡）

> 来源归档（dataset）

- **标题：** LET: Full-Size Humanoid Robot Real-World Dataset
- **类型：** dataset / humanoid / manipulation / rosbag
- **Hugging Face：** <https://huggingface.co/datasets/LejuRobotics/LET-Base-Dataset>
- **其它托管（数据卡互指）：** ModelScope `LejuRobotics/let_dataset`；AtomGit OpenLET/LET-Base-Dataset；引用块还写 `LET_Base_Dataset` 别名
- **组织：** LejuRobotics（乐聚）
- **许可：** **CC BY-NC-SA-4.0**（非商业、相同方式共享）
- **门控：** HF **ungated**（截至 2026-08-17）
- **入库日期：** 2026-08-17
- **一句话说明：** Kuavo 4 Pro（及轮式变体）真机多场景操作集：多视角 RGB-D、关节、夹爪/灵巧手，带原子技能时间轴标注；面向 IL / VLA 后训练。

## 访问与规模快照（截至 2026-08-17）

| 项 | 内容 |
|----|------|
| HF downloads | **63,482** |
| HF `usedStorage` | 约 **32 TiB**（含 LFS 历史，不等于「净数据体积」） |
| 文件数 | **26,384** siblings |
| rosbag | **25,824** 个 `.bag` |
| 同名 JSON 标注 | **511**（Labelled 轨迹级） |
| HDF5 | 本 HF 快照 **0** 个 `.h5/.hdf5`（数据卡仍描述 hdf5 目录树 → **文档超前于该镜像**） |
| 门控 | 无；数据卡另给申请邮箱 `wangsong@lejurobot.com` |
| 最后修改 | 2026-04-15 |

数据卡宣传口径（与文件计数独立，勿混用）：

| 字段 | 数据卡主张 |
|------|------------|
| 时长 | **>1000 小时**，持续更新 |
| 原子技能 | **117**（抓取、双手、工具使用等） |
| 子任务场景 | **31** |
| 领域 | 工业 / 家居 / 医疗 / 服务等 |
| 本体 | **Kuavo 4 Pro**（1.66 m / 55 kg / 40 DoF / 7 km/h；数据卡另提轮式版） |
| 标注 | 专家标注 + 人工核验；`marks[].skillAtomic`（如 `pick`）+ 中英 `skillDetail` |

Labelled 示例任务：快递扫码称重入库（`single_scan_code_for_weighing`）、线圈分拣、酒店送水/房卡、垃圾分类、流水线分拣等；Unlabelled 覆盖更多 3C/物流/桌面整理。采集地点元数据可见「长三角一体化示范区智能机器人训练中心」。

## 模态与格式

- **rosbag：** 头/左腕/右腕压缩 RGB 与深度；`/kuavo_arm_traj`（14 臂关节）；`/sensors_data_raw`（下肢 12 + 臂 14 + 头 2 + IMU）；灵巧手 `/control_robot_hand_position`、`/dexhand/state`；夹爪 `/leju_claw_*`；仿真夹爪 `/gripper/*`。
- **标签 JSON：** 场景三级编码、任务码、设备 SN（如 `P4-202`）、时间轴 `marks`（`startPosition`/`endPosition` 为归一化进度）。字段名保留官方拼写 **`loaction`**（location 笔误）。
- **工具链：** 数据卡指向 [kuavo_data_challenge](https://github.com/LejuRobotics/kuavo_data_challenge) 的 rosbag→LeRobot；产品化训练更应对 [LeTools-Learning](../repos/letools-learning.md)。

## 局限（归档时核实）

- **非商业许可**；ShareAlike 对衍生数据集有约束。
- HF 当前以 **rosbag 为主**；不要假设可直接 `load_dataset` 成 DataFrame（viewer 曾因 schema 漂移报 `CastError`）。
- 与 OpenLET AtomGit 旗舰仓是 **同一产品族的不同镜像**，文件集合与更新节奏可能不一致。

## 关联资料

- 社区策展：[openlet-let-base-dataset.md](../repos/openlet-let-base-dataset.md)（AtomGit）
- 产品站：[letools-lejurobot.md](../sites/letools-lejurobot.md)
- Wiki：[wiki/entities/let-base-dataset.md](../../wiki/entities/let-base-dataset.md)
