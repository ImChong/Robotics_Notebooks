# SPL Hockey Goalie Dataset：冰球守门员三维动作与冰球跟踪

> 一手资料归档（数据集；2026-11-19 核对，官方 README 标记 Last updated: November 2026）

- **标题：** SPL Hockey Goalie Dataset Documentation
- **类型：** dataset / markerless-motion-capture / sports-biomechanics / 3d-pose / puck-tracking
- **机构：** 枫叶体育娱乐集团（Maple Leaf Sports & Entertainment, MLSE）Sport Performance Lab；冰球跟踪来自 Stathletes 计算机视觉模型
- **官方原文：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/hockey/goalie/README.md
- **数据目录：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/tree/main/hockey/goalie/data/2026-08-06/P0001
- **原始样本：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/hockey/goalie/data/2026-08-06/P0001/HK_GL_P0001_S0001.json
- **可视化代码：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/hockey/goalie/animate.py
- **赛事文档：** https://stathletes.com/big-data-cup/data-documentation
- **许可：** CC BY-NC-SA 4.0（仓库 README / LICENSE；归属署名、非商业、相同方式共享）
- **入库日期：** 2026-11-19
- **核对范围：** 通读官方 hockey/goalie/README.md 全文（结构、字段、session、坐标系、19 关键点、平滑与对齐、动画示例）；读取仓库 README 的许可说明；读取 animate.py 全部函数与调用流程；解析 S0001 JSON 元数据与第一帧，核查公开目录含 81 个 shot JSON；核对 Big Data Cup 2027 官方赛事文档。
- **通读：** 已完成
- **一句话说明：** 一个守门员、一个采集 session 的 81 次射门，逐帧提供 19 点人体三维姿态以及冰球三维位置和速度，标称均为 60 Hz。
- **沉淀：** [SPL Hockey Goalie 3D 实体页](../../wiki/entities/spl-hockey-goalie-3d.md)

## 数据口径

| 字段 / 属性 | 已核实说明 |
|---|---|
| Session | 2026-08-06；1 名匿名守门员 P0001；81 shots（S0001–S0081） |
| `metadata.duration__s` | 单段时长，秒；S0001 为 7.333 s |
| `metadata.goalie_sampling_rate__hz` / `puck_sampling_rate__hz` | 均为 60 |
| `tracking[].frame` / `time` | 从 0 计数的帧索引；`time` 为该 shot 起点后的毫秒数 |
| `tracking[].data.goalie` | 19 个命名关键点到三维 `[x,y,z]`（英尺）；可能为 `null` |
| `tracking[].data.puck` | 冰球中心三维位置（英尺）；不可见时为 `null` |
| `tracking[].data.puck_velocity` | 冰球三维速度（英尺/秒）；缺失时为 `null` |
| 同步规则 | 每个人体帧匹配时间差 ≤8 ms 的最近冰球测量；**不插值** |
| 坐标系 | 冰场中心为原点，冰面 z=0；射门面向正 x 侧球门，球门线 x=89 ft |

19 点包括 NOSE、LEFT/RIGHT_EYE、LEFT/RIGHT_EAR、NECK、LEFT/RIGHT_SHOULDER、ELBOW、WRIST、HIP、KNEE、ANKLE，以及 MID_HIP。**“19 关节”并不精确**，眼鼻耳是人体标记点，不代表 19 个物理关节/自由度。守门员数据来自无标记动作捕捉，关键点经 3 次多项式、11 帧窗口的 robust Savitzky–Golay 平滑。

## 样本与对齐边界

- 已实测原始 `HK_GL_P0001_S0001.json`：`metadata.duration__s=7.333`，`tracking` 438 帧；首帧 `frame=0, time=0`，有 `puck`、`puck_velocity` 和 19 项 `goalie`。
- 所有 81 个 JSON 文件已在仓库目录列出；不把赛事页面“full dataset forthcoming”的旧文案误判为当前完全不可下载。
- 冰球仅在摄像机检测到时有观测值；遮挡/出画导致 `null`。60 Hz 是采样频率，不保证每一帧的冰球位置/速度均有效。
- 没有公开标定精度、全量误差分布、身体关节角度/力矩或可直接驱动机器人执行的动作标签。不能把此集当成机器人可直接模仿学习的 policy-action 数据。
- 原文未证实“全球首个冰球守门员 3D 数据集”，本归档不作首创性断言。

## 原始运行入口与复现提示

从 `hockey/goalie/` 工作目录执行官方 README 示例：先安装 `mplhockey`，再 `from animate import animate_shot`，传入 `data/2026-08-06/P0001/HK_GL_P0001_S0001.json`；函数返回 Matplotlib `FuncAnimation`。若需要保存视频，`notebook_mode=False` 后调用 `anim.save("S0001.mp4", writer="ffmpeg", fps=60)`，另外需要 ffmpeg。

`animate.py` 内部读取 JSON → 按关键点收集时间序列 → 将缺失项转为 NaN → 用 `MID_HIP` 跟随或固定镜头绘制关节线与冰球轨迹 → `FuncAnimation(update, frames=n_frames, interval=1000/sampling_rate)`。此代码是**离线动画回放**，不是运动估计模型的训练或实时推理入口；动画没有使用 `puck_velocity` 字段。

## 与库内知识的关系

- [实体页](../../wiki/entities/spl-hockey-goalie-3d.md)
- [具身数据清洗](../../wiki/concepts/embodied-data-cleaning.md)：比较时间对齐、缺失值与质量口径。
- [Roboflow Sports](../../wiki/entities/roboflow-sports.md)：体育视觉工具 vs 已发布的结构化三维轨迹数据集。
