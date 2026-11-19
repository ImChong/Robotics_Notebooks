# SPL Open Data（Sport Performance Lab 官方数据仓库）

> 一手官方仓库与代码归档（2026-11-19）

- **标题：** SPL Open Data
- **类型：** repo / dataset / sports-biomechanics / 3d-pose-tracking
- **机构：** 枫叶体育娱乐集团（Maple Leaf Sports & Entertainment, MLSE）Sport Performance Lab
- **代码与数据：** https://github.com/Sport-Performance-Lab/SPL-Open-Data
- **冰球数据子目录：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/tree/main/hockey/goalie
- **原始文档：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/hockey/goalie/README.md
- **动画源码：** https://github.com/Sport-Performance-Lab/SPL-Open-Data/blob/main/hockey/goalie/animate.py
- **赛事页面：** https://stathletes.com/big-data-cup
- **许可：** CC BY-NC-SA 4.0（数据与仓库说明；非商业条款须遵守）
- **核对范围：** 通读项目仓库 README、冰球 goalie README 与 animate.py 全文；读取 S0001 JSON；列出 session 目录确认存在 81 段 JSON；检查 Stathletes Big Data Cup 官方页面。
- **通读：** 已完成
- **开源状态：** 已公开 81 条 shot JSON 与可运行的 Python 3D 动画绘制工具；未发布内部无标记姿态估计 / 冰球检测模型的训练或推理代码。
- **入库日期：** 2026-11-19

## 实际文件与运行路径

| 文件 | 作用 |
|---|---|
| `hockey/goalie/README.md` | 数据文件命名、schema、单位、缺失值与对齐规则、动画示例 |
| `hockey/goalie/data/2026-08-06/P0001/HK_GL_P0001_S0001.json` 等 81 文件 | 逐帧 `metadata` 与 `tracking`，包括守门员 19 个 keypoint、冰球位置与速度 |
| `hockey/goalie/animate.py` | 离线动画 `animate_shot(path_to_json, ...)`；返回 `matplotlib.animation.FuncAnimation` |
| `hockey/goalie/assets/` | 坐标示意图、动画样例与多视角定向视频 |

README 入口：`pip install mplhockey` 后，在 `hockey/goalie/` 下执行 `from animate import animate_shot` 与 `animate_shot("data/2026-08-06/P0001/HK_GL_P0001_S0001.json")`。此例依赖 NumPy、Matplotlib；输出视频还需 ffmpeg。代码直接读现成 JSON，**不运行姿态估计/冰球检测**，并把缺失三维点变为 NaN 后不绘制。

## 已发布与未发布的边界

- **已发布：** 一名守门员一个 session 的 81 个 JSON、文档、示例动画源码，足以开展离线球–人体轨迹关联分析。
- **未随仓库发布：** 运动捕捉与冰球识别上游训练模型、底层标定系统、真实机器人执行动作/控制命令；不能据动画脚本推断其存在。
- **许可边界：** 仓库 CC BY-NC-SA 4.0，禁止未经授权的商业重用；自行检查后续授权变更与第三方依赖许可。
- **赛事页面差异：** Stathletes 文档仍有 “full dataset release is forthcoming”，以当前官方 GitHub 目录和可读取 JSON 作为可获取状态的直接证据。

## 对 wiki 的映射

- [SPL Hockey Goalie 3D 实体](../../wiki/entities/spl-hockey-goalie-3d.md)
- [SPL Hockey Goalie 数据集字段归档](../datasets/spl-hockey-goalie-3d.md)
- [Big Data Cup 2027 页面归档](../sites/stathletes-big-data-cup-2027.md)
