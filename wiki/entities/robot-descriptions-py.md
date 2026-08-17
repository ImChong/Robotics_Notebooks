---
type: entity
tags: [sim2real, tooling, urdf, mjcf, pinocchio, mujoco, pybullet, repo, inria]
status: complete
updated: 2026-08-17
summary: "robot_descriptions.py 把 190+ 开源 URDF/MJCF 变成可 pip 安装的 Python 模块：首次 import 下载缓存，再经 Pinocchio/MuJoCo/PyBullet 等 loader 得到可运行模型；包是 Apache-2.0，机型许可证逐条看 README。"
related:
  - ./awesome-robot-descriptions.md
  - ../comparisons/robot-description-catalogs.md
  - ../concepts/urdf-robot-description.md
  - ./pinocchio.md
  - ./mujoco.md
  - ./fiveages-sim-robot-descriptions.md
  - ./urdf-files-dataset.md
  - ./humanoid-motion-intelligence.md
  - ../queries/pinocchio-quick-start.md
sources:
  - ../../sources/repos/robot-descriptions-py.md
  - ../../sources/repos/awesome-robot-descriptions.md
  - ../../sources/repos/mujoco-menagerie.md
---

# robot_descriptions.py

[robot_descriptions.py](https://github.com/robot-descriptions/robot_descriptions.py) 是把分散在各 git 仓的 **URDF / MJCF** 收成 **Python 模块** 的加载层。HMI [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)把它放在「工程与实机部署」：解决的是实验脚本里 **资产 URL 与版本漂移**，不是再发明一种描述格式。

## 一句话定义

`pip install robot_descriptions` 之后，用名字加载 190+ 开源机器人描述：第一次 import 自动下载并缓存，然后交给 [Pinocchio](./pinocchio.md) / [MuJoCo](./mujoco.md) / PyBullet / iDynTree / yourdfpy 得到可计算或可仿真的模型。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| URDF | Unified Robot Description Format | 本包多数 `*_description` 模块的主格式 |
| MJCF | MuJoCo XML Format | `*_mj_description` 模块；大量来自 Menagerie |
| SRDF | Semantic Robot Description Format | 部分臂带规划侧语义，与 URDF 成对出现 |
| CLI | Command Line Interface | `uvx robot_descriptions pull / show_in_*` |
| API | Application Programming Interface | `loaders.pinocchio` 等后端适配 |

## 核心信息

| 字段 | 内容 |
|------|------|
| 机构 | 法国国家信息与自动化研究所（INRIA）谱系；GitHub org `robot-descriptions`；CITATION 第一作者 Stéphane Caron |
| 版本 | PyPI **3.1.0**（2026-08-17 核查） |
| 许可 | 包 **Apache-2.0**；机型文件 **逐条上游许可证** |
| 开源 | **已开源、可运行**（PyPI / conda-forge / `uvx`） |
| 规模 | README 宣称 **190+** 描述；分类含臂、双足、人形、四足、移动操作、无人机等 |

## 为什么重要

- **实验可复现的第一公里：** 论文脚本写死某 commit 的 URDF 路径会烂；这里用模块名 + 缓存，把「去哪下、下哪一版」收口。
- **后端一致入口：** 同一机型可走 Pinocchio 做动力学、MuJoCo 做接触、PyBullet 做快速可视化，而不手写六套路径逻辑。
- **发现层已策展：** 覆盖 [Awesome Robot Descriptions](./awesome-robot-descriptions.md) 中的大多数条目，并声称能在对应后端加载成功。

## 核心原理

### 加载路径

```mermaid
flowchart LR
  name["模块名\n如 go2_description"] --> cache["本地 cache\n首次 git 拉取"]
  cache --> paths["URDF_PATH / MJCF_PATH\nPACKAGE_PATH"]
  paths --> loader["loaders.pinocchio\nmujoco / pybullet / ..."]
  loader --> obj["Robot / MjModel / ..."]
```

两条 API：

1. **Loader：** `from robot_descriptions.loaders.pinocchio import load_robot_description` → `load_robot_description("upkie_description")`。
2. **子模块：** `from robot_descriptions import go2_description`，读 `URDF_PATH` 等常量。声明了 `XACRO_PATH` 时，包会 **透明展开 xacro** 再缓存 URDF。

CLI：`uvx robot_descriptions pull iiwa14_description`，以及 `show_in_mujoco` / `show_in_meshcat` / `show_in_pybullet` / `show_in_yourdfpy`。

### 命名习惯

| 后缀 | 含义 |
|------|------|
| `*_description` | 通常 URDF（偶带 SRDF） |
| `*_mj_description` | MJCF，常镜像 [MuJoCo Menagerie](https://github.com/google-deepmind/mujoco_menagerie) |
| `*_official_description` | 厂商/官方 ROS 2 描述（如 UR、TIAGo） |

## 工程实践

1. **先装包再选后端额外依赖：** `pip install robot_descriptions` 不够跑 MuJoCo loader，还需本机 MuJoCo Python 绑定。
2. **许可证逐行读：** 同表里 UR 新机型网格可能是 **厂商图形文档条款**；Stretch / iCub / GENE.01 等有 **CC-BY-NC** 标记。不要假设「包是 Apache 则模型可商用」。
3. **动力学标定另做：** 能加载 ≠ 惯量/摩擦已对齐真机。见 [URDF](../concepts/urdf-robot-description.md) 与 SysID 页。
4. **中国新机优先对照另一仓：** Galaxea R1 Pro 等已出现在本包，但 EngineAI / 多数轮式人形仍更全的是 [fiveages-sim](./fiveages-sim-robot-descriptions.md)。
5. **C++：** 需要时用第三方 [robot_descriptions.cpp](https://github.com/mayataka/robot_descriptions.cpp) 调本包，而非自己重新镜像资产。

| 检查项 | 建议 |
|--------|------|
| 安装 | `pip` / `conda-forge` / `uvx` |
| 后端 | Pinocchio、MuJoCo、PyBullet、iDynTree、yourdfpy、RoboMeshCat |
| 真机入口 | 本包 **不提供** SDK；只给描述文件路径 |

源码运行时序图对论文训练仓才强制；本页是库加载器，上图已覆盖运行时数据流。

## 局限与风险

- **不自托管网格：** 上游仓搬迁或删 tag 会导致首次 cache 失败；CI 绿不代表你克隆日仍可下。
- **不是 ROS 2 工作空间：** 要 `robot_state_publisher` + ros2_control，看 [fiveages-sim](./fiveages-sim-robot-descriptions.md) 或厂商 `*_description` 包。
- **不是 URDF 语料库：** 做 parser 回归、跨源 diff 应走冻结的 [URDF Files Dataset](./urdf-files-dataset.md)。
- **HMI 主表只是策展摘要：** 细节以 README 与上游 LICENSE 为准。

## 关联页面

- [Awesome Robot Descriptions](./awesome-robot-descriptions.md)
- [机器人描述目录选型](../comparisons/robot-description-catalogs.md)
- [URDF](../concepts/urdf-robot-description.md)
- [Pinocchio](./pinocchio.md) — `loaders.pinocchio` 的计算后端
- [Pinocchio 快速上手](../queries/pinocchio-quick-start.md)
- [MuJoCo](./mujoco.md) — MJCF / Menagerie 上游
- [fiveages-sim robot_descriptions](./fiveages-sim-robot-descriptions.md)
- [URDF Files Dataset](./urdf-files-dataset.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)

## 参考来源

- [robot_descriptions.py 来源归档](../../sources/repos/robot-descriptions-py.md)
- [Awesome Robot Descriptions 归档](../../sources/repos/awesome-robot-descriptions.md)
- [MuJoCo Menagerie 归档](../../sources/repos/mujoco-menagerie.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- 官方仓库：<https://github.com/robot-descriptions/robot_descriptions.py>
- PyPI：<https://pypi.org/project/robot_descriptions/>
- 贡献指南：<https://github.com/robot-descriptions/robot_descriptions.py/blob/main/CONTRIBUTING.md>
