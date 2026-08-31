# safeflow_arxiv_2603_23983

> 来源归档（ingest）

- **标题：** SafeFlow: Real-Time Text-Driven Humanoid Whole-Body Control via Physics-Guided Rectified Flow and Selective Safety Gating
- **类型：** paper
- **来源：** arXiv:2603.23983（2026-03-25 预印本）
- **作者：** Hanbyel Cho, Sang-Hun Kim, Jeonguk Kang, Donghan Koo
- **机构：** 三星电子（Samsung Electronics）Future Robot AI Group
- **入库日期：** 2026-08-31
- **最后更新：** 2026-08-31
- **项目页：** <https://hanbyelcho.info/safeflow/>
- **一句话说明：** 实时文本驱动人形全身控制：VAE 潜空间 **物理引导整流流匹配** + **Reflow** 单步采样（NFE=1），叠加 **训练无关三阶段安全门**（语义 OOD / 生成不稳定 / 硬运动学筛查），在 Unitree G1 上相对 TextOp 显著提升可执行性与成功率。

## 核心论文摘录（MVP）

### 1) 问题与总贡献（Abstract / §I）

- **链接：** <https://arxiv.org/abs/2603.23983>
- **痛点：** 纯运动学文本驱动生成器（如 TextOp）易产生 **物理幻觉**——关节越界、自碰撞、平衡失稳；开放域 / OOD 文本下更严重，下游运动跟踪控制器难以兜底。
- **SafeFlow 主张：** 两层架构——高层 **物理引导整流流** 生成可跟踪参考轨迹；部署时 **三阶段选择性安全门** 在语义、生成稳定性、运动学三层筛掉不安全输出，否则触发站立 fallback。
- **对 wiki 的映射：**
  - [SafeFlow 实体](../../wiki/entities/paper-loco-manip-161-104-safeflow.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [TextOp 实体](../../wiki/entities/paper-loco-manip-161-022-textop.md)（基线对照）

### 2) 物理引导整流流生成（§III-B）

- **生成器：** 在 VAE 潜空间做 **Rectified Flow Matching**；采样时注入与真机执行相关的物理目标（关节可行性、自碰撞回避、稳定性、运动平滑）。
- **实时化：** **Reflow** 蒸馏把物理引导内化进单步流，**NFE=1**；生成器单独约 **92.6 Hz**，完整安全管线约 **67.7 Hz**。
- **流式接口：** 与 TextOp 同范式——每步接收当前文本 \(l_t\) 与历史参考 \(T_{\mathrm{hist}}=2\)，输出未来 \(T_{\mathrm{fut}}=8\) 帧参考，再由低层 RL 跟踪控制器 \(\pi\) 转关节指令。
- **对 wiki 的映射：**
  - [π₀ Policy / Flow Matching](../../wiki/methods/π0-policy.md)（流匹配动作生成对照）
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 3) 三阶段安全门（§III-C，训练无关）

| 阶段 | 检测对象 | 机制 |
|------|----------|------|
| Stage 1 | 语义 OOD 文本 | 文本嵌入空间 **Mahalanobis** 分数 |
| Stage 2 | 生成不稳定 | **方向敏感性差异** 指标 \(\mathcal{R}\)；超阈则注入站立 prompt 并插值到预定义站姿 |
| Stage 3 | 硬运动学违规 | 关节 / 速度极限等最后一道筛查 |

- **对 wiki 的映射：**
  - [SafeFlow 实体](../../wiki/entities/paper-loco-manip-161-104-safeflow.md) §工程实践

### 4) 低层跟踪（§III-D）

- RL **运动跟踪控制器** 在仿真中训练，将接受的 kinematic 参考转为可执行关节命令；与高层生成解耦，但受益于更可执行的参考分布。
- **对 wiki 的映射：**
  - [Unitree G1](../../wiki/entities/unitree-g1.md)
  - [Reinforcement Learning](../../wiki/methods/reinforcement-learning.md)

### 5) 实验数字（§IV / 项目页）

| 方法 | JV ↓ | SC ↓ | Succ. ↑ | Empjpe ↓ |
|------|------|------|---------|----------|
| TextOp（基线） | 43.14% | 11.05% | 80.6% | 81.42 |
| SafeFlow (+ Guid. & Reflow) | **3.08%** | **1.42%** | **98.5%** | **40.89** |

| 管线 | 延迟 (ms) | 频率 (Hz) |
|------|-----------|-----------|
| TextOp 生成器 | 23.59 | 42.4 |
| SafeFlow (+ Guid. & Reflow) 生成器 | 10.80 | 92.6 |
| + 完整三阶段安全门 | 14.78 | **67.7** |

- **真机：** Unitree G1 长时域多行为（挥手、出拳、下蹲、单腿跳等）；高风险 prompt（如 "double backflip"）被安全门拦截并 fallback 站立。
- **对 wiki 的映射：**
  - [loco-manip-161-category-04](../../wiki/overview/loco-manip-161-category-04-generative-language-trajectory.md)

### 6) 开源核查（步骤 2.5，2026-08-31）

| 项 | 状态 |
|----|------|
| 项目页 | <https://hanbyelcho.info/safeflow/> — 摘要、方法图、量化表、真机视频齐全 |
| Code / GitHub | **未列出** — 页内无官方仓库、Hugging Face 或权重链接 |
| 结论 | **未开源** — 截至入库日无可辨识训练/推理/部署入口；源码运行时序图不适用 |

## 其他公开资料

- **项目页归档：** [sites/safeflow-hanbyelcho.md](../sites/safeflow-hanbyelcho.md)
- **161 策展槽位：** [loco_manip_161_survey_104_safeflow.md](loco_manip_161_survey_104_safeflow.md)
- **arXiv HTML：** <https://arxiv.org/html/2603.23983>

## 当前提炼状态

- 已升格完整论文实体页（保留 Loco-Manip 161 #104 文件名以稳定 catalog）。
