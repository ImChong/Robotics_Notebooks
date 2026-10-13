# VICAR: Video Imitation with Contact-Aware Retargeting for Humanoid Interactive Skills

> 来源归档（论文摘录；基于 arXiv v1，不是逐字全文）

- **类型：** paper（humanoid video imitation / contact-aware retargeting / loco-manipulation）
- **状态：** arXiv preprint v1，2026-10-09 提交
- **作者：** Dvij Kalaria, Bike Zhang, S. Shankar Sastry
- **机构：** University of California, Berkeley
- **论文：** <https://arxiv.org/abs/2610.13079>
- **HTML：** <https://arxiv.org/html/2610.13079v1>
- **项目页：** <https://ember-center-berkeley.github.io/VICAR/>
- **代码/项目仓库：** <https://github.com/ember-center-berkeley/VICAR>
- **许可证：** arXiv 页面标注论文 CC BY 4.0；不能据此推断所有代码/资产同一许可。
- **入库日期：** 2026-10-13
- **沉淀到 wiki：** [VICAR 项目实体](../../wiki/entities/paper-vicar-contact-aware-video-imitation.md)

## 摘要与贡献边界

VICAR 将第三人称人体视频恢复成可训练的人形机器人交互技能。系统恢复人体动作与任务相关障碍物，人工标记 task-defining contacts，再以两阶段轨迹优化保持接触世界位姿、末端速度与障碍 clearance；优化结果经 movable-contact augmentation 训练动作生成器和 Isaac Lab/PPO G1 参考跟踪策略。论文展示 10 项技能，部分在 Unitree G1 真机上测试。

本文贡献的中心在**接触感知重定向与跨接触位置动作生成**。它并未提出新的全局场景理解器、纯视觉硬件策略或力控接触模型；运行时仍借助外部 motion capture 更新全局球/物体位置。

## 方法摘录

### A. 人体动作与接触标签

GENMO/GEM 从单目第三人称视频恢复人体姿态、全局 root 与任务关键点。接触集合写作：

`\mathcal{A}=\{(\tau_k,b_k,\hat{p}_k,\hat{R}_k)\}_{k=1}^K`

其中 $\tau_k$ 是接触帧、$b_k$ 是目标机器人链接，$(\hat p_k,\hat R_k)$ 是世界系接触位姿。该定义是**指定时空位置的 pose objective**，不是对接触力、摩擦锥、滑动或接触模式建模。乒乓球例子标注抛球与球拍击球位姿，击球点可在推理时跟随估计球位置移动。接触帧/部位/姿态当前为人工标签，约 20–30 分钟/技能。

### B. 与任务相关的场景障碍

采用 UniDepth 逐帧深度、Semantic-SAM 语义分割，并用 GPT-4o mini 从候选区域中选出可能限制动作的桌面、桌沿、地面等。通过人体尺度估算恢复 metric scale，再将深度像素反投影并变成占据几何。论文使用 5 cm 体素、至少 5 个点作为占据门槛。目标是重建对机器人运动有约束的局部场景，而非完整扫描环境。

### C. 两阶段重定向目标

1. **初值：** OmniRetarget 风格 interaction mesh / Laplacian keypoint loss + 时间正则化 + 关节位置/速度限制；此阶段不启用障碍惩罚。
2. **整段精化：** 对浮动基座与全身关节序列同时优化，目标含位置速度保持、旋转变化保持、平滑性、障碍物穿透惩罚和 anchor pose 残差：

`\mathcal{L}=\mathcal{L}_{p}+\lambda_R\mathcal{L}_R+\lambda_s\mathcal{L}_s+
\lambda_{col}\mathcal{L}_{col}+\lambda_a\mathcal{L}_{anchor}`

接触/避碰以高权重 penalty 优先，但不是不可违反的硬等式。若 anchor 和可达几何冲突，算法寻找近似可行解并保留 residual 指标；关节角、速度另有约束。

### D. 条件生成与 PPO tracking

优化轨迹形成 motion family。动作生成器输入 motion style、movable contact conditions 与已有时序上下文，输出未来一段机器人/物体参考运动。发球通过周围三维网格改变 hit point；拾取改变目标 grasp pose；双手搬箱约束同步抓取；梯子保留序列化手/脚支撑点。

跟踪策略在 Isaac Lab 的 Unitree G1 上训练：

- 仿真步长 0.005 s，action decimation 4，名义策略频率 50 Hz；测试 25 Hz variant。
- 每轮 4,096 并行环境，PPO 最多 30,000 iterations。
- 观测包括 reference、anchor 位姿、相对 body/joint 状态、base 线/角速度和上一动作；动作是各关节位置目标。
- 随机化摩擦、恢复系数、关节默认位置、躯干质心，并每 1–3 秒施加随机推扰。
- 奖励跟踪 anchor/body pose 与 velocity，并惩罚动作变化、软关节限位和非预期接触。

## 技能集与定量结果

十个技能由六种乒乓球 serve、桌面 pickup、桌下 pickup、bimanual box pickup/place 和 ladder climbing 构成。比较对象 PHC、GMR、VideoMimic、OmniRetarget；各基线都在统一 PPO、reward、DR 与训练步数下评估。论文给出的成功率为仿真 1,000 次 trial：

| 技能 | VICAR | OmniRetarget | 解读 |
|---|---:|---:|---|
| Table-tennis serves | 99.8% | 99.8% | 接近持平；VICAR 是 99.8%，GMR 97.2% |
| Tabletop pickup | 99.8% | 76.2% | 接触/路径连续性影响显著 |
| Under-table pickup | 99.2% | 98.1% | 两者均高，VICAR 速度保持误差更小 |
| Bimanual pick-place | 100% | 100% | 成功率相同 |
| Ladder climbing | 95.5% | 56.0% | 仿真差异大，但未转化为真机爬梯成功 |

真机报告：乒乓球发球 21/30（70%）；桌面拾取 10/10（100%）；桌下拾取 7/10（70%）；双手搬箱 6/10（60%）。梯子仅有仿真结果，论文说明真机盲跟踪器会绊倒。所谓 zero-shot sim-to-real 是“未用真机训练样本”，并非视觉端到端部署：policy 读取本体感知与 projected gravity，外部 motion capture 提供全局物体/球状态。Dex3-1 手用于拾取。

## 论文定位与局限

- 相比 PHC/GMR 的关键点/骨架匹配，VICAR 把全局任务 anchor、障碍几何和相对末端速度纳入整段轨迹优化。
- 相比 OmniRetarget 的 interaction-mesh 与硬约束，VICAR突出世界系 contact pose penalty 和速度保持；并非所有任务都有成功率增益。
- 人工接触标注、任务相关 scene reconstruction、运动捕捉工程成本高；每种新任务家族需要单独标注与场景处理。
- unreachable contact、平衡支撑不足、手部能力限制都可能失败；爬梯 sim2real 结果明确暴露盲 tracking 限制。
- 单目深度、语义分割与 GPT 选区错误的鲁棒性不是本文研究贡献。
- penalty 不等于严格接触约束；没有显式 force/contact mode/sliding 模型。
- 论文分析训练源的外部仓库与公开项目网站时要区分：网页 GitHub 仓库公开了项目页和可视化/数据准备工具，原始训练、refinement、augmentation 脚本仍在 TT_PLayer checkout。

## 来源核对状态

- [x] 作者、机构、日期、摘要与 CC BY 4.0：arXiv abstract/HTML v1。
- [x] 方法公式、模型栈、任务数量与表 IV：arXiv HTML v1。
- [x] 真机成功率、MoCap 运行时依赖、爬梯失败与 limitations：arXiv HTML v1。
- [x] 仓库可复现范围：官方仓库 README、VISER.md、VISUALIZATION_SOURCES.md。
