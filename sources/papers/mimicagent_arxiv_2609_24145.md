# MimicAgent: Quadruped Skills via Prompt-to-Trajectory Generation（arXiv:2609.24145）

> 来源归档（ingest）

- **标题：** MimicAgent: Quadruped Skills via Prompt-to-Trajectory Generation
- **类型：** paper / quadruped / imitation-learning / example-guided-rl / llm-agents / sim2real
- **arXiv abs：** <https://arxiv.org/abs/2609.24145>
- **PDF：** <https://arxiv.org/pdf/2609.24145>
- **项目页：** <https://luckykantnayak.github.io/mimic-agent/> — 归档见 [`sources/sites/mimic-agent-github-io.md`](../sites/mimic-agent-github-io.md)
- **代码：** **待发布**（项目页标注 *code coming soon*，截至 2026-09-23 无 GitHub URL）
- **机构：** 卡内基梅隆大学（Carnegie Mellon University）— Lucky Kant Nayak*、Narayanan Palghat Parameswaran*、Neehar Peri、Deva Ramanan（* 共同一作）
- **入库日期：** 2026-09-23
- **一句话说明：** **Prompt-to-trajectory** agentic 管线：LLM  coding agent 从文本生成 **粗粒度参考轨迹**（MuJoCo FK，无物理仿真），再喂给 **example-guided RL**（DeepMimic 族）训练四足技能；Claude Fable 5.1 下 **87%** prompt 语义对齐，多技能上用户偏好常优于 Eureka 与手工 keyframe。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | <https://luckykantnayak.github.io/mimic-agent/> | 7 技能 baseline 对比、57 人用户研究、真机 Go2 / Go2-W |
| Eureka 对照 | [Eureka](https://arxiv.org/abs/2310.12931) | LLM 自动 reward design；本文认为其跨技能/形态泛化弱 |
| Example-guided RL | [DeepMimic](https://arxiv.org/abs/1804.00657) | 参考轨迹 + RL 跟踪 |
| 人形 mocap 对照 | AMASS / OMOMO 等 | 四足缺大规模公开参考动作库 |

## 摘要级要点

- **问题：** 四足动态技能 RL 依赖 **reward shaping**（「graduate student descent」）；Eureka 等 LLM reward 设计在 **多样技能与形态**（含 wheeled quadruped）上 brittle。
- **关键观察：** 对人/LLM 而言，**生成参考运动** 比 **设计 reward** 更容易；粗且 **动力学不可行** 的轨迹仍可作为 example-guided RL 的有效目标。
- **MimicAgent 管线：** (1) **Motion planner agent** 将 prompt 扩写为分阶段技能描述；(2) **Code generator** 输出 base/foot 轨迹可执行代码；(3) **Kinematic executor**（MuJoCo FK，无 physics）；(4) **Task-agnostic unit tests**（高度包络、关节限位、穿地、非 aerial 脚接触）；(5) **Diagnostic + code revision** 自改进环（最多 3 轮）。
- **下游 RL：** example-guided RL 将动力学可行性交给 RL；支持 **Unitree Go2** 与 **Go2-W** 真机（trot、bound、flip、skating、handstand 等）。
- **扩展：** 同管线可生成 **SMPL 人形** 日常与 acrobatic 参考（walk/run/jump/cartwheel 等），但论文主实验为四足。

## 核心摘录（面向 wiki 编译）

### 1) 参考轨迹表示

相位变量 $\phi \in [0,1]$，关键帧 $s_t = [p_t^{\mathrm{base}}, \alpha_t^{\mathrm{base}}, \theta_t]$（位置、四元数、关节角）。

### 2) Unit tests（技能无关）

- Base height envelope
- Joint limit violation count
- Ground penetration（四足全穿地重罚）
- Feet air-time（非 aerial 技能至少半周期有脚接地）

### 3) 评测技能（7）

Trot、Bound、Side Flip、Front Flip、Aerial Crossover、Crab Diagonal Scuttle、Reverberating Yaw Pulse — Go2 / Go2-W 分工见项目页。

### 4) 开源状态（项目页，2026-09-23）

| 组件 | 状态 |
|------|------|
| 项目页 / 视频 | 已公开 |
| 代码仓库 | **待发布**（code coming soon） |
| 预训练 checkpoint | 未列链接 |

## 对 wiki 的映射

- 新建：[paper-mimicagent](../../wiki/entities/paper-mimicagent.md)
- 交叉：[reinforcement-learning](../../wiki/methods/reinforcement-learning.md)、[hybrid-locomotion](../../wiki/tasks/hybrid-locomotion.md)

## 当前提炼状态

- [x] arXiv + 项目页核查
- [x] 开源状态：待发布
- [ ] 代码发布后补 `sources/repos/` 与时序图
