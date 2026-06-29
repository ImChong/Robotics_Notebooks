# CHORD: Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration

> 来源归档（ingest）

- **标题：** Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration（CHORD）
- **类型：** paper / tech-report / dexterous-manipulation / contact-wrench / imitation-rl / benchmark / video-to-data
- **机构：** NVIDIA
- **作者：** Xinghao Zhu*, ‡, Zixi Liu*, Shalin Jain*, Chenran Li†, Milad Noori†, Huihua Zhao, John Welsh, Michael Andres Lin, Wei Liu, Tingwu Wang, Xingye Da, Zhengyi Luo, Vishal Kulkarni, Naema Bhatti, Yuke Zhu, Linxi Fan, Bowen Wen, Danfei Xu, Soha Pouya, Yan Chang‡（* equal；† core；‡ lead & corresponding）
- **原始链接：**
  - 项目页：<https://nvidia-isaac.github.io/video_to_data/chord/>
  - Tech report PDF：<https://nvidia-isaac.github.io/video_to_data/chord/chord.pdf>
  - V2D 文档：<https://nvidia-isaac.github.io/video_to_data/>
  - 代码：<https://github.com/nvidia-isaac/video_to_data>
- **入库日期：** 2026-06-29
- **一句话说明：** 用 **物体中心接触力旋量空间（CWS）奖励** 把人类双手演示迁移到灵巧手 RL 策略——比较接触对物体产生的力学效应而非仅匹配接触位置；联合 imitation + task tracking + VOC 课程，在 **4,739** 项双手 benchmark 上训练，**1,831** 项评测平均成功率 **82.12%**，并扩展到全身 G1+Dex3（**90.77%**）与真机 Sharpa 双手开环/闭环部署。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| V2D 管线 | <https://github.com/nvidia-isaac/video_to_data> | Video Ingestion → Reconstruction → Robotic Grounding 三阶段 |
| 位置引导基线 | DexMachina [1] | 演示接触位置 + VOC 虚拟物体控制 |
| 力邻近基线 | ManipTrans [2] | 演示手-物交互附近的接触力奖励 |
| 物理重定向基线 | SPIDER [3] | 接触位置 + 课程式虚拟接触力（arXiv:2511.09484） |
| 仿真训练 | Isaac Lab [42] | benchmark 环境与策略训练宿主 |
| 真机 | Dexmate + 双 Sharpa 手 | mocap 跟踪位姿；开环 action chunk / 闭环推理 |
| 全身 | Unitree G1 + Dex3 三指 | 手部-only / 第三人称演示经 inpainting 扩展全身 |

## 摘要级要点

- **问题：** 人类演示丰富，但灵巧手 **具身差异** 使回放人手轨迹不可行；现有方法要么依赖脆弱的演示迁移假设，要么需要逐任务对齐的人-机数据，难以规模化。
- **核心洞察：** 接触是人与机器人的桥梁——应匹配接触对物体诱导的 **瞬时运动**，而非 3D 接触位置；用 **物体中心 wrench 空间** 表征各接触配置的力-矩方向与可支持物体运动。
- **CHORD 框架：** 给定人类参考 $\tau^{\text{ref}}$（手关键点 + 物体部件位姿），IK retarget 到机器人 $\mathbf{x}^{\text{robot}}$；策略 $\pi(a_t \mid o^{\text{robot}}_t, o^{\text{object}}_t; \mathbf{x}^{\text{robot}}_t, \mathbf{x}^{\text{object}}_t)$ 以 rollout 物体位姿跟踪参考为目标，奖励 $r = r_{\text{task}} + r_{\text{imit}} + r_{\text{contact}}$，并沿用 DexMachina 的 **VOC（Virtual Object Controller）** 课程。
- **CWS 奖励：** 对每个物体部件 $k$，从演示提取接触位置/法向，用摩擦锥多面体近似构造 primitive wrench 矩阵 $\mathcal{W}^{h,k}$；用 **support function** $\sigma^{h,k} = \max_{\text{col}} \mathcal{B}^\top \mathcal{W}^{h,k}$ 比较人与机器人 wrench 几何，得 $r_{\text{cws}}^k$（相对容差 $\beta$ 的双侧指数核）；并惩罚 **unintended / missed** 接触。
- **Position vs Wrench（§3.1 案例）：** 同一物体区域、不同法向/力方向可产生截然不同的物体运动；位置奖励会选中「空间近但物理错」的接触。
- **训练稳健化：** 沿参考轨迹任意状态 reset + VOC 稳定窗；从 $\mathcal{W}^{h,k}$ 采样扰动物体；残差动作空间 + retarget 先验；VOC 课程退火。视频重建噪声大时退化为 **force-closure basis** 正支撑奖励 $r_{\text{fc}}^k$。
- **全身扩展：** 手部-only 演示 → inpainting 预测全身；第三人称全身演示但手指重建噪声大 → 用 $r_{\text{fc}}^k$ 替代直接 wrench 匹配。
- **Benchmark（§3.3）：** 整合开源动捕数据集 [13–16, 39–41] + 自研视频重建；**4,739** 可仿真双手任务（刚体/关节体/多物体）；相对 DexMachina / ManipTrans / Spider 在 **horizon、contact events、Ferrari-Canny epsilon** 上更密更长。

## 核心摘录（面向 wiki 编译）

### 1) 奖励分解

| 项 | 形式 | 作用 |
|----|------|------|
| $r_{\text{task}}$ | 各部件 SE(3) 跟踪 + 多物体相对位姿项 $r_{\text{relative}}$ | 物体运动跟随演示 |
| $r_{\text{imit}}$ | 机器人状态相对 retarget 参考 | 正则化到可行机器人轨迹 |
| $r_{\text{contact}}$ | CWS + unintended/missed 惩罚 | 接触力学效应对齐 |
| VOC | 辅助 wrench 沿参考推物体 | 早期探索稠密信号；需课程退火防局部最优 |

### 2) 大规模评测（§4.1）

- **1,831** 任务、**同一套超参**（VOC 增益、课程、奖励权重）；作者称首个在此规模评测的 RL 灵巧操作方法。
- **成功判据：** 物体位置误差 >15 cm 或旋转误差 >40° 则终止；completion ratio >0.7 为任务成功。
- **平均成功率：82.12%**（Figure 5 左）。

### 3) 与基线对比（Table 1 摘要）

| 套件 | 指标 | 基线 | 基线分 | CHORD |
|------|------|------|--------|-------|
| DM (DexMachina) | AUC | DexMachina | 0.232±0.214 | **0.687±0.358** |
| MT (ManipTrans) | MT-SR | ManipTrans | 0.428 | **0.639** |
| SP (Spider) | SP-SR | Spider | 0.333±0.488 | **0.359±0.482** |
| Ours-1 | AUC | DexMachina | 0.211±0.138 | **0.895±0.052** |
| Ours-1 | SP-SR | Spider | 0.133±0.327 | **0.999±0.000** |
| Ours-2 | SP-SR | Spider | 0.533±0.503 | **0.982±0.022** |

（各行基线套件与协议不同，分数仅行内可比。）

### 4) 消融与相关性（§4.2–4.3）

- **Full CWS vs Position Only vs No Contact：** CWS 显著优于仅位置（DexMachina 式）与无接触项。
- **CWS reward ↔ 成功率：** 1,831 runs 上 Pearson **r ≈ 0.80**（分数据集 **0.76–0.89**）；单调饱和，可作为训练信号与代理指标。
- **长时程跟踪：** DexMachina ADD-AUC 下 CHORD 在 ~48 s 序列仍保持高物体跟踪精度，基线随 horizon 退化。

### 5) 全身与真机（§4.5–4.6）

- **全身：** 手部-only 与第三人称演示 → G1 + Dex3；**90.77%** 成功率。
- **真机：** Dexmate + 双 Sharpa；mocap 位姿；开环 action chunk 与闭环均成功；抬箱等任务对接触时序敏感。

### 6) 局限（§5 摘要）

- 依赖仿真资产质量与接触估计；视频重建演示需退化为 force-closure 目标。
- 尚未覆盖 vision-based 部署、更噪演示鲁棒性、任务感知成功度量。

## 对 wiki 的映射

- 沉淀实体页：[CHORD（接触力旋量引导灵巧操作）](../../wiki/entities/paper-chord-contact-wrench-dexterous-manipulation.md)
- 交叉更新：[contact-rich-manipulation.md](../../wiki/concepts/contact-rich-manipulation.md)、[manipulation.md](../../wiki/tasks/manipulation.md)、[spider-physics-informed-dexterous-retargeting.md](../../wiki/methods/spider-physics-informed-dexterous-retargeting.md)、[dexterous-manipulation-data-pipeline.md](../../wiki/queries/dexterous-manipulation-data-pipeline.md)、[isaac-lab.md](../../wiki/entities/isaac-lab.md)

## 引用（Tech Report）

```bibtex
@techreport{zhu2026chord,
  title={Learning Dexterous Manipulation Using Contact Wrench Guidance From Human Demonstration},
  author={Zhu, Xinghao and Liu, Zixi and Jain, Shalin and others},
  institution={NVIDIA},
  year={2026},
  url={https://nvidia-isaac.github.io/video_to_data/chord/chord.pdf}
}
```
