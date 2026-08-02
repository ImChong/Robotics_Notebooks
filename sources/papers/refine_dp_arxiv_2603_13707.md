# refine_dp_arxiv_2603_13707

> 来源归档（ingest）

- **标题：** REFINE-DP: Diffusion Policy Fine-tuning for Humanoid Loco-manipulation via Reinforcement Learning
- **类型：** paper
- **来源：** arXiv:2603.13707（2026-03-14 预印；IEEE RA-L：2026-03-01 收稿 / 05-31 修回 / 07-01 接收）
- **作者：** Zhaoyuan Gu*, Yipu Chen*, Zimeng Chai*, Alfred Cueva, Thong Nguyen†, Yifan Wu†, Huishu Xue†, Minji Kim, Isaac Legene, Fukang Liu, KyoungMok Kim, Ayan Barula, Yongxin Chen, Ye Zhao（*† equal contribution）
- **机构：** 佐治亚理工学院（Georgia Tech）Institute for Robotics and Intelligent Machines（IRIM）
- **入库日期：** 2026-08-02
- **最后更新：** 2026-08-02
- **项目页：** <https://refine-dp.github.io/REFINE-DP/>
- **一句话说明：** 分层 DP 运动规划器 + RL loco-manip 跟踪器；以 DPPO/PPO **联合微调** 缩小规划–控制分布错配，Booster T1 上实现开门/搬箱等长程 loco-manip，仿真 90%+ SR、约 20× 降示教量。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2603.13707>
- **核心贡献：** 人形 loco-manipulation 上，离线训好的扩散规划器与低层控制器解耦 → 跟踪差、分布漂移、长程失败；扩示教对高维人形过贵。REFINE-DP 用 **层次框架**（DP 输出紧凑笛卡尔动作块：基座速度 + 双手 SE(3)；RL 控制器转关节参考）并 **联合优化** 两者：DP 用 PPO 式扩散策略梯度（DPPO）抬任务成功率，控制器同步适应规划器演化中的命令分布。
- **对 wiki 的映射：**
  - [REFINE-DP 实体](../../wiki/entities/paper-loco-manip-161-157-refine-dp.md)
  - [Diffusion Policy](../../wiki/methods/diffusion-policy.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)

### 2) 低层控制器与采数（§III-A）

- **解耦上下身：** 下身 **足端落点跟踪**（非纯速度跟踪）+ 上身双手 SE(3) 跟踪；Isaac Lab 训；动作相对默认姿态的关节位置偏移，PD 跟踪。
- **采数：** VR 遥操作（约 50 条核心行为含恢复）+ 任务启发式 planner 扩到约 1000 条成功轨迹；冻结低层控制器执行。
- **对 wiki 的映射：**
  - [Isaac Lab](../../wiki/entities/isaac-lab.md)
  - [Teleoperation](../../wiki/tasks/teleoperation.md)

### 3) DP 预训练与 DPPO 微调（§III-B/C）

- **预训练：** 标准 Diffusion Policy 噪声预测损失；观测 horizon 8、动作 chunk 12、0.1 s 步。
- **微调：** 把去噪步嵌入增广 MDP（DPPO），对 DP 做 PPO；稀疏任务奖励即可从约 50–70% 预训练 SR 抬到 90%+。
- **对 wiki 的映射：**
  - [Policy Optimization / PPO](../../wiki/methods/policy-optimization.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 4) 联合优化（§III-D / Alg.1）

- **交替：** rollout → PPO 更新低层 → 再 rollout → DPPO 更新 DP（外环约 L=2）。
- **动机：** 控制器预训练时命令是独立采样的静止目标；DP 命令是连续轨迹上的移动目标 → 分布错配。联合优化把控制器暴露在规划器命令分布下。
- **对 wiki 的映射：**
  - [Residual Policy Learning](../../wiki/methods/residual-policy-learning.md)（对照：残差修正 vs 直接调 DP 参数）

### 5) 实验数字（§IV）

| 设定 | 关键读数 |
|------|----------|
| 仿真 SR | 预训练约 **50–70%** → REFINE-DP **>90%**（每任务 100 trials） |
| 数据效率 | 纯预训练达 90% ≈ **1000** 轨迹；**50** 轨迹 + 微调 → **95–97%** |
| 联合优化 | 只调低层：长程 pick-place **+18%** SR；达 90% 迭代 **40→20**；朝向误差可 **−50%**，EE 速度约 **−15%** |
| OOD 初始化 | 最大随机化下预训练 **0%** → 课程微调 **>80%** |
| 真机（N=20） | Task1 **70%** / Task2 **50%** / Task3 **75%**；吞吐约 **+10% / +20%**（拾箱/开门） |
| 平台 | Booster T1 29 DoF；DP 10 Hz TensorRT，低层 50 Hz；MoCap 或 RealSense+AprilTag |

### 6) 开源核查（步骤 2.5，2026-08-02）

| 项 | 状态 |
|----|------|
| 项目页 | <https://refine-dp.github.io/REFINE-DP/> — 方法/视频/附录 reward 表齐全 |
| Code 按钮 | **无链接**（`<span class="link-btn">`，非 `<a href>`） |
| GitHub | `REFINE-DP/REFINE-DP` 仅为站点源（`gh-pages`） |
| 结论 | **未开源** — 无可辨识训练/推理/部署入口；源码运行时序图不适用 |

## 其他公开资料

- **项目页归档：** [sites/refine-dp-github-io.md](../sites/refine-dp-github-io.md)
- **161 策展槽位：** [loco_manip_161_survey_157_refine-dp.md](loco_manip_161_survey_157_refine-dp.md)
- **arXiv HTML：** <https://arxiv.org/html/2603.13707>

## 当前提炼状态

- 已升格完整论文实体页（保留 Loco-Manip 161 #157 文件名以稳定 catalog）。
