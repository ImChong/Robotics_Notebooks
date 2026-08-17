# ICRA 2026 REAL-I Challenge 数据集（kuavo_data_challenge_icra）

> 来源归档（dataset / challenge）

- **标题：** [ICRA 2026] The 1st Real-World Embodied-AI Learning Challenge Datasets
- **类型：** dataset / benchmark / competition
- **Hugging Face：** <https://huggingface.co/datasets/LejuRobotics/kuavo_data_challenge_icra>
- **赛事：** REAL-I（Real-World Embodied-AI Learning Challenge），ICRA 2026，乐聚主办
- **本体：** Kuavo 4 Pro
- **许可：** HF 卡 **无 license 元数据**（截至 2026-08-17）；使用前核对赛事手册
- **门控：** **ungated**
- **入库日期：** 2026-08-17
- **一句话说明：** 工业操作挑战赛数据包：仿真三任务各约 1000 条 rosbag；真机三任务目录声明但 **real 尚未更新**；另有未写入 README 的 `vienna/` 附加包。

## 访问快照（截至 2026-08-17）

| 项 | 内容 |
|----|------|
| HF downloads | **2,395** |
| `usedStorage` | 约 **1.49 TiB**（页面曾显示 1.63 TB） |
| siblings | **3,406** |
| `.bag` | **3,401** |
| YAML card | 空（HF 警告 missing yaml metadata） |
| 最后修改 | 2026-06-02 |
| Viewer | 仅文档配图（imagefolder 误标 size &lt;1K）；**不要**用 Datasets viewer 当训练入口 |

## 任务与计分（数据卡）

仿真 / 真机各三任务；仿真满分 100，超时每秒 -1。

| 轨道 | 任务 | 内容 |
|------|------|------|
| Sim | TASK1 Toy Sorting | 桌上随机玩具：动物→右篮、车→左篮；起始位姿/桌高/摆放随机 |
| Sim | TASK2 Parcel Weighing | 传送带软包 → 电子秤 → 另一传送带；秤与包裹位姿随机 |
| Sim | TASK3 Conveyor Belt Sorting | 传送带随机朝向工业件，成功分拣 4 件 |
| Real | TASK1 Rubbish Sorting | 可回收→蓝箱、其它→灰箱（布局随机） |
| Real | TASK2/3 | 与仿真称重、传送带分拣同构 |

**目录声明：** `sim/` 三任务已给；`real/` **not updated yet**。

**HF 实际 bag 计数：**

| 前缀 | bags |
|------|------|
| `sim/TASK1-ToySorting` | 1000 |
| `sim/TASK2-ParcelWeighing` | 1000 |
| `sim/TASK3-ConveyorBeltSorting` | 1000 |
| `vienna/bottle` | 123 |
| `vienna/express` | 158 |
| `vienna/parts` | 120 |

`vienna/` **未出现在数据卡目录树**；当作附加/未文档化子集，赛规以手册为准。

## 格式

全部为 **rosbag**，保留原始传感器；训练需自行解析或走 [LeTools-Learning](../repos/letools-learning.md) / [kuavo_data_challenge](https://github.com/LejuRobotics/kuavo_data_challenge) 转换。

## 硬件口径（数据卡，Kuavo 4 Pro）

身高 **1.66 m**、体重 **55 kg**、热插拔电池；**40 DoF**、最高行走 **7 km/h**、双足自主 SLAM；宣传支持多模态大模型与 **20+ 原子技能**。

## 对 wiki 的映射

- 升格：[wiki/entities/icra-2026-real-i.md](../../wiki/entities/icra-2026-real-i.md)
- 社区背景：[wiki/entities/openlet.md](../../wiki/entities/openlet.md)
- 更大规模真机集：[let-base-dataset.md](let-base-dataset.md)
