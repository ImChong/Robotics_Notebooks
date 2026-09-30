# HOI-Retarget: Contact-Centric Retargeting for Human-Object Interaction（arXiv:2609.34674）

> 来源归档（ingest）

- **标题：** HOI-Retarget: Contact-Centric Retargeting for Human-Object Interaction
- **类型：** paper / humanoid / motion-retargeting / loco-manipulation / hoi / dataset
- **arXiv abs：** <https://arxiv.org/abs/2609.34674>
- **PDF：** <https://arxiv.org/pdf/2609.34674>
- **项目页：** <https://shinben0327.github.io/hoi-retarget/> — 归档见 [`sources/sites/hoi-retarget-shinben0327-github-io.md`](../sites/hoi-retarget-shinben0327-github-io.md)
- **代码：** **已开源** — <https://github.com/shinben0327/hoi-retarget>（BSD-3-Clause）；归档见 [`sources/repos/hoi-retarget.md`](../repos/hoi-retarget.md)
- **数据集：** <https://huggingface.co/datasets/shinben0327/hoi-retarget>；3D 预览 Space：<https://huggingface.co/spaces/shinben0327/hoi-retarget-viewer>
- **机构：** 苏黎世联邦理工学院机器人系统实验室（ETH RSL）— Jihwan Shin、Adrià López Escoriza、Junzhe He、Matthias Heyrman、Marco Hutter
- **入库日期：** 2026-09-30
- **一句话说明：** **以物体坐标系接触点为目标** 的窗口化轨迹优化，把 SMPL-X HOI 转为 G1/H2 可学参考；OMOMO 上 mean contact gap **18.3 cm→0.5 cm**（vs OmniRetarget），**4.6×** 更快；发布 **6,952 clips / 13.8 h / 75 objects**。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| GitHub | <https://github.com/shinben0327/hoi-retarget> | `hoi-retarget` CLI；Pinocchio + CasADi/IPOPT |
| HF Dataset | <https://huggingface.co/datasets/shinben0327/hoi-retarget> | 重定向后机器人 HOI |
| 对照 | OmniRetarget、GMR、DynaRetarget | 交互 mesh vs 接触点；动力学 refinement 下游 |
| 源数据 | OMOMO、ParaHome、NeuralDome、CoRoleHOI、IMHD² | 五源 + 双机协作 |

## 摘要级要点

- **问题：** LfD 扩展到人形 **loco-manipulation** 缺 **机器人可用 HOI 参考**；纯 pose 对齐会把手移离物体或换到错误表面。
- **阶段 1（III-A）：** GMR 系 IK + **身高比缩放物体 mesh/轨迹**；接触目标 \(p^{o*}_{c,i,t}\) 锚在 **物体系** → 尺度增广时 target 随表面走。
- **阶段 2（III-B）：** 重叠窗口 NLP（\(H\) 帧、前 \(p\) 帧 pin 上一窗）；代价：跟踪 IK、**接触位置/手掌朝向**、脚 stance、jerk；关节界与速度界。
- **III-C：** Viser 交互工具修正单目重建的 **接触段** 再优化。
- **扩展：** 多机器人共操纵同一物体轨迹；CARI4D 单目桌搬运；DynaRetarget / RL tracker 作 **dynamic refinement** 初始化对比。

## 核心摘录（面向 wiki 编译）

### 1) OMOMO / G1（13 类物体，相对 OmniRetarget）

| 指标 | OmniRetarget | HOI-Retarget |
|------|--------------|--------------|
| Mean contact-point gap | 18.3 cm | **0.5 cm** |
| Retarget 速度 | 1× | **~4.6×** |

### 2) 发布数据集

- **6,952** robot HOI clips，**13.8 h**，**75** unique objects，**G1 + H2**
- 源：OMOMO、ParaHome、NeuralDome、CoRoleHOI、IMHD²（HUMOTO 展示但不在 release）

### 3) 能力表（相对 GMR / OmniRetarget / DynaRetarget）

- **独有强调：** 显式 **Contact Location Preservation**、**Multi-Agent** 源、**Temporal Coupling**（窗口 NLP）
- **动力学：** 本方法 kinematic；输出可接 DynaRetarget / SBTO / RL tracker

## 对 wiki 的映射

- 新建：[paper-hoi-retarget](../../wiki/entities/paper-hoi-retarget.md)
- 交叉：[paper-hrl-stack-03-omniretarget](../../wiki/entities/paper-hrl-stack-03-omniretarget.md)、[motion-retargeting-gmr](../../wiki/methods/motion-retargeting-gmr.md)、[loco-manipulation](../../wiki/tasks/loco-manipulation.md)、[paper-notebook-dynaretarget-dynamically-feasible-retargeting-us](../../wiki/entities/paper-notebook-dynaretarget-dynamically-feasible-retargeting-us.md)

## 当前提炼状态

- [x] arXiv + 项目页 + GitHub + HF 核查（2026-09-30）
- [x] 开源：代码 BSD-3 + 数据集 HF；SMPL-X / 源 motion 需自备许可下载
- [x] 源码运行时序图（对齐 README 模块与 CLI）
