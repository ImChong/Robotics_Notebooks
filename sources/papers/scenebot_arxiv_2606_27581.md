# SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction

> 来源归档（ingest）

- **标题：** SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction
- **类型：** paper / humanoid / motion-tracking / contact-conditioning / loco-manipulation / scene-reconstruction / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2606.27581>
- **项目页：** <https://ericcsr.github.io/scenebot/>（归档见 [`sources/sites/scenebot-ericcsr-github-io.md`](../sites/scenebot-ericcsr-github-io.md)）
- **机构：** Amazon FAR（Frontier AI & Robotics）、Stanford University、CMU；† Amazon FAR team co-lead
- **作者：** Sirui Chen, Shibo Zhao, Zhen Wu, Jiaman Li, Guanya Shi†, C. Karen Liu†
- **硬件：** Unitree G1；机载 LiDAR + 骨盆 IMU；笔记本（i5-13200H + RTX 3050）经以太网跑策略与状态估计
- **仿真：** MuJoCo sim-to-sim 评测；PPO 训练 contact-aware whole-body controller
- **入库日期：** 2026-06-29
- **一句话说明：** 用 **per-link contact label** 把自由空间 locomotion、地形穿越与全身物体操作统一到 **单一 motion tracking policy**；以 **hindsight scene reconstruction** 从人体/机器人运动反推场景交互图并合成地形与物体资产，在 **7.5 小时** 重建数据上训练；自由空间性能与 **SONIC** 相当，场景交互任务显著优于 SONIC 等通用 tracker。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://ericcsr.github.io/scenebot/> | 浏览器 MuJoCo 交互 demo、单策略多场景真机视频 |
| 对照基线 | [SONIC（arXiv:2511.07820）](https://arxiv.org/abs/2511.07820) | 自由空间强、场景交互近零成功率（论文 Table 3） |
| 场景数据对照 | [OmniRetarget（arXiv:2509.26633）](https://arxiv.org/abs/2509.26633) | scene-aware retarget vs hindsight reconstruction |
| 同类接触跟踪 | [CHIP（arXiv:2512.14689）](https://arxiv.org/abs/2512.14689)、[WoCoCo（arXiv:2406.06005）](https://arxiv.org/abs/2406.06005) | 论文 Table 1 横向对比 |
| 状态估计 | [SuperOdometry（IROS 2021）](https://arxiv.org/abs/2104.06135) | LiDAR-IMU 根位姿；与骨盆 IMU 融合 |

> 截至入库时，论文与项目页标注 **Code / Data Coming Soon**。

## 摘要级要点

- **问题：** 现有 RL motion tracker 在 **自由空间** 表现强，但纯运动学跟踪无法消解 **物体/地形接触** 的物理歧义（手腕靠近箱子可能是伸手、轻触或施力搬起；脚靠近台阶边缘不等于产生足够 GRF 上台阶）。
- **SceneBot 接口：** 在参考运动之外，高层只需额外给出 **哪些 robot link 应与哪类 scene（terrain / object）建立接触**；标签只定义在机器人本体，不绑定具体物体几何，从而保持 **低层 tracking policy** 定位而非 vision-based task policy。
- **数据引擎：** **Hindsight scene reconstruction** — 对 retarget 后机器人运动，先建 **robot-scene interaction graph**（关键 link：双腕、双脚、骨盆；低相对速度/加速度候选边 + 碰撞/力闭合剪枝），再重建 **2.5D 高程地形** 与 **平行板物体** 资产。
- **训练：** PPO；输入 $(c_t, g_t, s_t, p_{\text{root,t}})$；在 BeyondMimic/SONIC 类 tracking reward 上叠加 **contact correctness** 与 **contact duration**；平地无标签数据上 terrain contact reward **仅在 $c=1$ 时激活**，部署时 contact label 置零可回退自由空间跟踪。
- **部署：** **SuperOdometry** + 骨盆 IMU 融合估计根位姿/线速度（200 Hz）；真机 mocap 对照平均位置误差 **0.032 m**、姿态 **0.018 rad**、线速度 **0.092 m/s**。
- **数据：** AMASS、OMOMO、Bones、LAFAN 混合；约 **7.5 小时** 重建 contact-rich 数据。
- **亮点任务：** 搬箱上楼、坐椅、踢腿、跑酷式地形、双手搬大箱等 **长时程 terrain + object** 组合；**全部 demo 单策略**。

## 核心摘录（面向 wiki 编译）

### 1) Contact conditioning 与策略接口

- **策略：** $\pi(c_t, g_t, s_t, p_{\text{root,t}}) \rightarrow a_t$；$g_t \in \mathbb{R}^{93}$ 含下身关节角速度、头/腕 6D pose、根位姿误差等（遵循 BeyondMimic/CHIP 约定）；$c_t \in \{0,1\}^n$ 为 link×{terrain, object} 二值接触意图。
- **设计取舍：** 不要求感知物体 ID 或地形 mesh；高层用 **接触意图** 消解运动学歧义，比端到端 vision task policy 更轻、更可组合。

### 2) Hindsight scene reconstruction 两阶段

| 阶段 | 内容 |
|------|------|
| Interaction graph | 异构图：robot nodes $\mathcal{K}$={L/R wrist, L/R foot, pelvis} + scene type nodes {terrain, object}；时序边记录 $(t_s, link, scene\_type, p)$ |
| Scene assets | 地形：2.5D elevation，接触边处加 plateau 后合并/挖碰撞；物体：与接触面平行的 plate 集合，轨迹由 grasp 边均值定 |

- **相对 OmniRetarget：** 论文假设 reconstruction **保证场景可支撑机器人运动**，而非优先服从已有场景约束；OMOMO 上训练 **reconstruction > tuned Omni-retarget**（物体-手对齐/穿透更少 → 抓取失败率更低）。

### 3) Contact-aware RL 奖励

- $r_{\text{cr}}$：desired vs actual contact label 一致（仅 $c_{\text{des}}=1$ 时计分）。
- $r_{\text{dr}}$：累计接触时长，clip 0.5 s 防止拒释物体。
- **Grasp 训练技巧：** 物体重建轨迹未必从地面静止开始 → 稳定力闭合前对物体施加 **heuristic stabilizing force（magic force）** 增密 contact reward；**contact-mismatch termination** 强制限时建立期望接触。

### 4) 消融与定量（MuJoCo sim-to-sim，各 20 条未见序列）

| 任务类 | Ours SR | SONIC SR | 要点 |
|--------|---------|----------|------|
| free-space | 100% | 100% | 关节/根跟踪误差与 SONIC 同级 |
| object | 95% | 5% | 无 hand label 训练 → 近零抓取 |
| terrain | 100% | 15% | 无 global root → terrain 45%、object 20% |
| sit | 100% | 0% | 骨盆-椅子接触 |

- **Global root 必要性：** 去掉 global position/velocity，terrain 成功率 **100%→45%**，object **95%→20%**；局部跟踪漂移导致 motion-terrain 错位（Fig. 6）。
- **测试时关 foot label：** 仍可达 terrain **85%**（训练无 foot label 仅 60%），说明 label 简化训练、部分平衡机制可保留。

### 5) 与代表性 whole-body controller 对比（论文 Table 1）

SceneBot 声称是首个在 **单策略** 下同时满足：**多样参考跟踪 + 地形交互 + 物体交互**（相对 BeyondMimic、SONIC、TWIST、Any2Track、OmniRetarget、CHIP、WoCoCo 等组合能力）。

### 6) 局限（论文 §5）

- 场景重建依赖 **高质量 retarget**；脚滑等差重定向会破坏 interaction graph。
- 视频/生成模型动作若物理不一致，会 destabilize graph 构建。

## 对 wiki 的映射

- 沉淀实体页：[SceneBot（arXiv:2606.27581）](../../wiki/entities/paper-scenebot.md)
- 交叉更新：[sonic-motion-tracking.md](../../wiki/methods/sonic-motion-tracking.md)、[humanoid-motion-tracking-method-selection.md](../../wiki/queries/humanoid-motion-tracking-method-selection.md)、[loco-manipulation.md](../../wiki/tasks/loco-manipulation.md)、[paper-loco-manip-161-114-omniretarget.md](../../wiki/entities/paper-loco-manip-161-114-omniretarget.md)

## 引用（arXiv）

```bibtex
@article{chen2026scenebot,
  title={SceneBot: Contact-Prompted General Humanoid Whole Body Tracking with Scene-Interaction},
  author={Chen, Sirui and Zhao, Shibo and Wu, Zhen and Li, Jiaman and Shi, Guanya and Liu, C. Karen},
  journal={arXiv preprint arXiv:2606.27581},
  year={2026}
}
```
