# Learning Diverse Humanoid Tasks via Synthetic Video Scenarios without Real World Data（arXiv:2607.21648）

> 来源归档（ingest）

- **标题：** Learning Diverse Humanoid Tasks via Synthetic Video Scenarios without Real World Data
- **类型：** paper / humanoid / synthetic-data / generated-video / motion-tracking / reinforcement-learning
- **arXiv abs：** <https://arxiv.org/abs/2607.21648>
- **PDF：** <https://arxiv.org/pdf/2607.21648>
- **HTML：** <https://arxiv.org/html/2607.21648>
- **DOI：** <https://doi.org/10.48550/arXiv.2607.21648>
- **机构：** 国立成功大学（National Cheng Kung University, NCKU）机械工程系
- **作者：** Yun-Hao Tsai、Cong-Thanh Vu、Yen-Chen Liu
- **发表 / 上传：** 2026-07-22（arXiv v1）
- **硬件 / 仿真：** Unitree G1；**Isaac Lab**；PPO；4096 并行 env（Xeon W5-3435X + RTX 4000 Ada）
- **视频生成：** Google **Veo 3 / 3.1** API（文内 Fig.1 写 3.1，实验节写 Veo 3 API）
- **重定向：** SMPL-X 估计 → **GMR** → IK 精修
- **项目页 / 代码：** 截至入库日 **无**（arXiv「Code, Data, Media」无官方仓；作者 GitHub `sean901109` 未见对应仓）
- **入库日期：** 2026-08-05
- **一句话说明：** 用 **文本提示 → 生成视频 → SMPL-X/GMR 参考 → motion stitching → DeepMimic 式 RL 跟踪**，在 **无真机/无 MoCap** 条件下于仿真学习多样人形任务风格。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 论文 | [arXiv:2607.21648](https://arxiv.org/abs/2607.21648) | 唯一官方入口 |
| 对照：零样本生成视频 HOI | [GenHOI](../../wiki/entities/paper-loco-manip-03-genhoi.md) | 单视频接触约束，非 RL 跟踪训策略 |
| 对照：仿真 teleop 数据 | [OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md) | 仍依赖操作员 VR，非纯生成视频 |
| 重定向工具 | [GMR](../../wiki/methods/motion-retargeting-gmr.md) | 本文引用 Araujo et al. arXiv:2510.02252 |
| 跟踪范式 | [DeepMimic](../../wiki/methods/deepmimic.md) | 模仿奖励与非严格根跟踪 |

## 摘要级要点

- **问题：** LfD / DeepMimic 类方法依赖昂贵 MoCap；同任务人类执行方式多样，单条示范覆盖不足。
- **方法：** 结构化提示约束 Veo 生成全身可见、静态相机视频 → ViT+SMPL-X 重建 → GMR 重定向 → **根坐标对齐 + 关节过渡缓冲** 拼接多段 → Isaac Lab 中 PPO 非对称 actor–critic 跟踪。
- **评测：** 50 个日常任务提示 × 每提示 10 条视频；策略侧展示 lie-and-stand、boxing、pick-and-place 等；关节位置 MAE 约 **0.04–0.07 m**；**0.5 kg** 负载下上身力矩增大、下身轨迹基本不变。
- **局限：** 后空翻等高动态段易时间不一致；结论明确 **未来工作含 sim-to-real**——当前证据主在仿真。
- **开源状态（截至 2026-08-05）：** **确认未开源**（无项目页、无 GitHub、论文未承诺 release）。

## 核心摘录（面向 wiki 编译）

### 1) 管线三阶段

1. **Video → humanoid reference：** 指令提示（全身可见、单人、静态相机等）+ 用户动作提示；SMPL-X；GMR 尺度/对齐/脚滑与穿地修正 + IK。
2. **Motion stitching：** 第二段根位姿对齐到第一段终点；插入短过渡缓冲做关节插值，避免断续。
3. **RL tracking：** 根相对观测；critic 特权含干净参考；奖励 = DeepMimic 式模仿项 − 限位/平滑/自碰惩罚；摩擦/恢复系数、关节偏置、躯干 CoM、速度扰动随机化。

### 2) 关键训练超参（文内经验值）

| 组 | 取值 |
|----|------|
| 模仿权重 | \(w_p=0.65, w_v=0.1, w_e=0.15, w_c=0.1\) |
| 指数尺度 | \(\alpha_p=2.0, \alpha_v=0.1, \alpha_e=40.0, \alpha_c=10.0\) |
| 正则 | \(w_{\mathrm{limit}}=1.0, w_{\mathrm{smooth}}=0.1, w_{\mathrm{contact}}=0.1\) |
| 网络 | MLP [512,256,128]，ELU；PPO；4096 agents |

### 3) 与邻近生成式路线对照

| 维度 | 本文 | GenHOI | OASIS |
|------|------|--------|-------|
| 视频角色 | **多样示范 → RL 跟踪训练** | 单视频 → 接触几何 → 零样本执行 | 非生成视频；仿真 VR teleop |
| 每任务策略 | 需训练 | 不训 task-specific 策略 | 训高层 FM + 低层 WBC |
| 真机 | 未报（sim-to-real 留待未来） | 有真机 | 有真机零样本 |
| 开源 | **无** | 见其项目页 | 已开源 |

## 对 wiki 的映射

- 沉淀实体页：[paper-synthetic-video-humanoid-tasks.md](../../wiki/entities/paper-synthetic-video-humanoid-tasks.md)
- 交叉补强：[GenHOI](../../wiki/entities/paper-loco-manip-03-genhoi.md)、[OASIS](../../wiki/entities/paper-loco-manip-04-oasis.md)、[Imagine2Real](../../wiki/entities/paper-imagine2real-zero-shot-hoi.md)、[loco-manip-category-02](../../wiki/overview/loco-manip-category-02-synthetic-data.md)、[GMR](../../wiki/methods/motion-retargeting-gmr.md)、[DeepMimic](../../wiki/methods/deepmimic.md)、[Unitree G1](../../wiki/entities/unitree-g1.md)

## 当前提炼状态

- [x] arXiv HTML/PDF 方法与评测摘录
- [x] 开源核查：无项目页 / 无代码（步骤 2.5）
- [x] 机构注册：`ncku` → 国立成功大学（National Cheng Kung University）
- [x] wiki 实体与生成式数据对照交叉规划
