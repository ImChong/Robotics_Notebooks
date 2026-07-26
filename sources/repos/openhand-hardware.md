# openhand-hardware

> 来源归档

- **标题：** openhand-hardware — Yale OpenHand CAD
- **类型：** repo
- **来源：** Yale Grab Lab（`grablab` GitHub 组织）
- **链接：** <https://github.com/grablab/openhand-hardware>
- **项目页：** <https://www.eng.yale.edu/grablab/openhand/>
- **License：** CC BY-NC 3.0（见 `LICENSE.md`；GitHub 显示 `Other` / NOASSERTION）
- **Stars / Forks：** ~230 / ~114（2026-07-26 快照）
- **默认分支：** `master`
- **最近推送：** 2025-03-25（含 Model F3 README 更新）
- **入库日期：** 2026-07-26
- **一句话说明：** OpenHand 各型号的 **SolidWorks + STL** CAD 仓库；打印件在对应文件夹 `.stl`，改型看 SolidWorks 装配；控制代码不在本仓，见 `openhand_node`。
- **沉淀到 wiki：** 是 → [`wiki/entities/yale-openhand.md`](../../wiki/entities/yale-openhand.md)

---

## 核心定位

本仓只承载 **机械设计文件**：各手型独立目录 + `fingers` / `couplings` / `common parts` 共享件。完整装配说明以 [OpenHand 网站](https://www.eng.yale.edu/grablab/openhand/) 为准；常用型号（T / T42 / O）的 Python/ROS 控制见 [`openhand_node`](https://github.com/grablab/openhand_node)。

零件命名惯例（Model Q 除外）：

| 前缀 | 含义 |
|------|------|
| `a*_handName` | 大结构件（自上而下） |
| `b*_handName` | 齿轮 / 舵机连接小件 |
| `c*_handName` | 手指安装座 |
| `d*_handName` | 可选件 |

多数手型提供 **flexure / pivot** 等多种指关节选项；flexure 常用 Smooth-On 聚氨酯浇注（切模或可复用多件模）。HDM 细节见项目页 [hdm.html](https://www.eng.yale.edu/grablab/openhand/hdm.html)。

## 仓库目录（master）

| 路径 | 内容 |
|------|------|
| `model t` / `model t42` / `model o` / `model q` / `model m2` / `model vf` | 经典欠驱动 / 多模态型号 |
| `stewart hand` / `sphinx hand` | 并联手内操作型号 |
| `model f3 (forces-for-free hand)` | **Model F3**：STL、SLDPRT、装配 PDF、手指装配 |
| `fingers` / `couplings` / `common parts` | 共享指、腕耦合、通用件 |
| `params_print.SLDPRT` | 打印参数相关零件 |
| `README.md` / `LICENSE.md` / `Banner.jpg` | 文档与许可 |

### Model F3 子目录要点

- `stl/`、`sldprt/`：打印与改型源文件
- `Model F3 Assembly Guide 1.0.pdf`：装配指南
- `F3_finger_assembly.SLDASM`：手指装配

## CAD 使用注意（README）

- SolidWorks 打开装配前将 **External References → Load referenced documents** 设为 **All**，避免缺依赖警告。
- 大量使用 **Configurations** 减少源文件数量。
- 更完整 CAD 工作流见项目页 [OpenHand CAD Guide (PDF)](https://www.eng.yale.edu/grablab/openhand/OpenHand%20CAD%20Guide.pdf)。

## 关联生态仓

| 仓库 | 角色 |
|------|------|
| [`openhand_node`](https://github.com/grablab/openhand_node) | 活跃控制（O / T / T42 + ROS；MIT） |
| [`openhand-software`](https://github.com/grablab/openhand-software) | **已弃用**（2019-09 起） |
| [`openhand_simulation`](https://github.com/grablab/openhand_simulation) | 仿真资源 |
| [`Yale-OpenHand-Workshop-2018`](https://github.com/grablab/Yale-OpenHand-Workshop-2018) | 2018 workshop 材料 |

## 对 wiki 的映射

- [Yale OpenHand](../../wiki/entities/yale-openhand.md)
- 交叉：[EN02-OP](../../wiki/entities/en02-op.md)、[Deimel 欠驱动柔顺手](../../wiki/entities/paper-deimel-compliant-underactuated-robotic-hand.md)、[Manipulation](../../wiki/tasks/manipulation.md)、[抓取专题](../../wiki/overview/topic-grasp.md)
