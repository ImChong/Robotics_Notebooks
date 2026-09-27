# Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection（arXiv:2607.14236）

> 来源归档（ingest）

- **标题：** Never Too Late for Force: Accelerating VLA Post-Training with Reactive Force Injection
- **缩写 / 框架：** **LIFT**（*L**ate Reactive **I**njection of **F**orce for VLA Post-**T**raining*）
- **类型：** paper / vla / force / post-training / online-dagger / pi05 / contact-rich / flexiv
- **会议：** CoRL 2026（项目页标注）
- **arXiv：** <https://arxiv.org/abs/2607.14236>（PDF：<https://arxiv.org/pdf/2607.14236>）
- **项目页：** <https://lift-policy.github.io/> — 归档见 [`sources/sites/lift-policy-github-io.md`](../sites/lift-policy-github-io.md)
- **代码：** <https://github.com/y-wng/lift> — 归档见 [`sources/repos/y-wng-lift.md`](../repos/y-wng-lift.md)
- **作者：** Yi Wang、Wendi Chen、Zimo Wen、Han Xue、Xueqi Li、Wenye Yu、Zhijie Chen、Hao Yang、Jun Lv、Chuan Wen、Cewu Lu 等（* 共一；‡ 项目负责人；† 通讯）
- **机构：** 上海交通大学（SJTU）、上海创智学院（Shanghai Innovation Institute）、南方科技大学（SUSTech）、上海交通大学致远学院、诺玛矩阵（Noematrix Ltd.）
- **入库日期：** 2026-09-27
- **一句话说明：** 在 **π₀.₅** 预训练 VLA 上做 **力感知后训练**：复制 action expert 为 **reactive 流**、**因果力记忆 + 零初始化 cross-attn** 实现 chunk 内刷新动作，且 **初始化输出与原策略等价**；**在线 DAgger** 混合 vision-only 离线对齐与 **Flexiv TDK 力启用人工纠错**，三任务上相对 **vision-only online DAgger** 学得更快、峰值与最终性能更高。

## 开源状态（步骤 2.5）

- **项目页核查（2026-09-27）：** 摘要写明 code publicly available；页脚链到 **GitHub `y-wng/lift`**。
- **仓库核查：** README 明确基于 **OpenPI**，提供 **离线训练 + 在线 DAgger 启动脚本 + 策略推理服务**；**不含** 真机驱动、TDK 采集栈、`nmx_nedf_api`（NEDF2→LeRobot 转换依赖外部 SDK）。
- **结论：** **部分开源** — 训练/推理/数据格式与 launcher **已发布**；完整真机闭环需 **Flexiv + TDK + 部署侧上传/驱动**。

## 摘录 1：问题与三目标（O1–O3）

- **痛点：** 预训练 VLA 强依赖视觉；进入 **接触态** 时遮挡、深度歧义、小力误差易把执行推出离线示范分布；力/力矩 **难在预训练规模** 同步采集（平台与末端差异大）。
- **O1 反应式力注入：** 最近 **6D 末端力** 经 **因果力记忆** 注入 reactive expert，**chunk 内** 刷新动作；VLM 前缀 **算一次并缓存**，力更新不必重跑整段 VLM。
- **O2 保留预训练先验：** 复制原 action expert 权重；**shifted causal attention** 使 reactive token 在初始化时等价于原 fully-attentive action token；力 cross-attn **零初始化输出投影** → 训练前 **力残差为 0**，输出与原 VLA 一致。
- **O3 异构数据：** vision-only batch **mask 力记忆**（力路径无梯度）；online correction batch 启用实测力；固定 **1:1 offline:online** 采样。

**对 wiki 的映射：** [`wiki/entities/paper-lift-reactive-force-vla-posttrain.md`](../../wiki/entities/paper-lift-reactive-force-vla-posttrain.md)；与 [ForceVLA](../../wiki/entities/paper-forcevla.md)（预训练期力 MoE）、[TACO](../../wiki/entities/paper-taco-tactile-wm-vla-posttrain.md)（WM 合成纠错数据）对照。

## 摘录 2：两阶段训练与 online DAgger

1. **Stage 1 — 视觉任务对齐：** 手持设备 **vision-only** 示范；训练时 **mask force**；把 π₀.₅ 对齐目标任务。
2. **Stage 2 — 真机接触适应：** Flexiv Rizon 4S + **6D 末端力**；Flexiv **TDK** 采集 **力启用人工纠错**；与离线集混合 **反复部署–收集–更新**，跟踪 **策略诱导的力分布偏移**（force OOD / covariate shift）。

**对 wiki 的映射：** 实体页画 **Recognize 式** 闭环改为 **DAgger 数据环**；强调与 **offline DAgger 固定 buffer**（三任务全线下于 online，book **0 分**）及 **residual policy** 基线对比。

## 摘录 3：实验设置与主要结论

| 轴 | 要点 |
|----|------|
| **平台** | Flexiv Rizon 4S；6D EE force；π₀.₅ 基座 |
| **任务** | 毛巾折叠（分级 0.25–1.0）、书本插入（0.5/1.0）、汉诺塔环放置（0/1） |
| **对比** | LIFT full；π₀.₅ + online DAgger（无力）；LIFT w/o reactive（单帧力）；LIFT offline DAgger；π₀.₅ offline handheld；residual policy |
| **评测** | 每 checkpoint **3×10** 自主 rollout（n=30），95% CI；shift 测试每条件 10 rollout |
| **预算** | 约 1000/2000 step 可比；单次 online 实验约 **2–3 h**、**20–30** episodes |
| **Q1** | 力启用后训练 **加速** 且 **峰值/最终** 高于 vision-only online DAgger |
| **Q2** | **Reactive 力历史** 在书插入/汉诺塔关键；毛巾单帧力也可，但仍优于纯视觉 |
| **Q3** | 最终 ckpt 在对象/桌布/光照 shift 下 **未见明显退化**（相对 in-distribution） |
| **Q4** | **Online DAgger** 必要；offline-only 力纠错 buffer 三任务均差，book **零分** |
| **Q5** | Residual 在弱 base + 稀疏干预下 **显著低于 LIFT** |
| **Q6** | 纯 online（0:1）毛巾 **远差于** 1:1/1:2；主实验固定 **1:1** |

## 摘录 4：与相关线区别（项目页 Related Work 摘要）

- **ForceVLA / TA-VLA / ForceVLA2 / FAVLA：** 多在 **架构期** 融合力/力矩；LIFT 强调 **late post-training**、复制权重 + 零初始化力路径 + **latency-aligned** 力历史。
- **RDP / ImplicitRDP：** slow-fast 与力记忆；LIFT 用 **shifted causal attention** 保留初始化 action 上下文。
- **DAgger / CR-DAgger：** 分布偏移与 residual 纠错；LIFT 从 **视觉示范 + 在线纠错学 full action**，非 frozen base + 小 residual  alone。

## 局限（论文/项目页）

- 人工纠错 **吞吐受限**；评测 **单臂** Flexiv；未来需减纠错负担并扩展臂/传感器/末端。
