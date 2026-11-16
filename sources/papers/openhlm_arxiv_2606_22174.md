# OpenHLM（arXiv:2606.22174）

> 来源归档（ingest）

- **标题：** OpenHLM: An Empirical Recipe for Whole-Body Humanoid Loco-Manipulation
- **缩写：** **OpenHLM**
- **类型：** paper / humanoid / loco-manipulation / VLA / teleoperation / co-training
- **arXiv：** <https://arxiv.org/abs/2606.22174>（v2，2026-11-02）
- **全文（HTML）：** <https://arxiv.org/html/2606.22174>
- **通读：** 已完成
- **通读范围：** v2 正文 §1–§6、Appendix A–E（任务/协议、硬件、实现、长程基线、额外实验）
- **通读记录日期：** 2026-11-11
- **项目页：** <https://openhlm-project.github.io/>
- **代码：** <https://github.com/OpenHLM-project/OpenHLM>
- **数据 / 权重：** HF `OpenHLM/OpenHLM-data` · `OpenHLM/OpenHLM-ckpts`
- **作者：** Yingdong Hu, Haodong Zhu, Boyuan Zheng, Yihang Hu, Tong Zhang, Zunhao Chen, Junming Zhao, Ruiqian Nai, Yang Gao
- **机构：** 清华大学；上海期智研究院；千寻智能（Spirit.AI）
- **首次发表：** 2026-06-20
- **核对版本：** arXiv:2606.22174v2（2026-11-02） · CoRL 2026 Spotlight（官方项目页/代码仓标注）
- **入库日期：** 2026-07-22
- **开源状态：** **已开源**（代码 + 数据 + checkpoint；项目页与 README 互链）
- **一句话说明：** 用 **单变量消融路线图** 回答「如何做全身原生人形 VLA」：关节级全身遥操作优于部分自由度接口；静态/轮式双臂预训练可迁移到人形全身动作空间；**HuMI** 异构共训扩展新物体/指令而无需额外全身遥操作；系统级长程任务以更少演示时长超过 GR00T N1.6 与 Ψ₀。

## 核心论文细读（v2，包含附录）

### 1) 问题与主张（Abstract）

- **痛点：** 多数人形系统把上下身解耦成「轮式双臂式」行为，无法协调整条运动链。
- **问法：** 构建 **语言+像素 → 全部自由度** 的 whole-body-native VLA，需要哪些经验决策？
- **答法：** 三阶段 one-variable-at-a-time：全身遥操作接口 → VLA 设计 → 异构共训 → 得到可复现 **OpenHLM recipe**。
- **对 wiki 的映射：**
  - [OpenHLM](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
  - [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md)
  - [VLA](../../wiki/methods/vla.md)

### 2) 遥操作接口（Phase 1）

- **对照：** Decoupled（21-D）/ VR 3-point（24-D）/ **Joint-based whole-body（32-D）**。
- **结论：** 仅关节全身接口能完成踩踏板、蹲身穿架等任务；后两阶段均建立在该接口上。
- **对 wiki 的映射：**
  - [Teleoperation](../../wiki/tasks/teleoperation.md)
  - [Whole-Body Control](../../wiki/concepts/whole-body-control.md)

### 3) VLA 设计与异构共训（Phase 2–3）

- **接口适配**（投影初始化、动作排序、相对/绝对、本体）单点翻转仅轻微掉点——**非瓶颈**。
- **预训练：** π₀.₅ 初始化 ~91% task progress ≫ PaliGemma ~60% ≫ random ~42%。
- **推理：** 多步 flow matching 优于 one-step / drifting（尽管后者验证 MSE 可能更低）。
- **共训：** 在 Tasks 1–8 全身遥操作基础上，对 held-out 9–11 加入 **HuMI** 或 **stationary teleop**，progress 36% → 84%/89%（oracle 全全身遥操作 96%）。
- **对 wiki 的映射：**
  - [Imitation Learning](../../wiki/methods/imitation-learning.md)
  - [GR00T-WholeBodyControl](../../wiki/entities/gr00t-wholebodycontrol.md) — 低层执行栈对照

### 4) 系统结果（项目页 / 论文）

- 长程果盘（低桌–中桌–高架、双手分拣、20 有序果对）：OpenHLM **87.5%** @ **1.14 h** demo vs GR00T N1.6 **57.5%** / Ψ₀ **48.8%** @ **2.70 h**。
- 12 任务平均 task progress **>90%**（四类：抓放+行走、工作空间扩展、身体当工具、环境约束）。

## v2 逐节通读与验证记录（2026-11-11）

> **证据边界：** 以下为对 [arXiv v2 HTML 全文](https://arxiv.org/html/2606.22174) 正文及附录的逐节核查摘要，并与 [官方项目页](https://openhlm-project.github.io/) 和 [代码 README](https://github.com/OpenHLM-project/OpenHLM) 对照；没有运行训练、复现真机或独立测出成功率。论文采用 task progress，不等于二值成功率。所有数字是作者报告的实机实验。

### §1–§2：设计目标与评测体系

- **G1** 单策略直接命令全身；**G2** 多任务语言控制且少样本；**G3** 低成本异构数据可扩展新物体、指令与动作。
- **HLM-12** 含 12 个语言条件任务，四类分别为行走抓放、全身工作空间拓展、非手肢体当操作器、环境接触/几何约束。Appendix A 定义逐任务评分 rubric：每个「策略 × 任务」在真机独立测试 **5 次**，使用相同 5 组起始场景；评分为 **progress fraction**，带标准误，不可将它写成最终成功率。

### §3.1、Appendix C.1/E.1：全身采集与低层接口

- 高层操作员/策略输出参考动作（章节默认典型 10 Hz）；SONIC/动作跟踪器低层约 50 Hz 输出关节目标，再由 PD 跟踪。
- 录制的身体动作 **32-D = root roll + root pitch + yaw rate（3）+ G1 29 个关节（臂 14、腿 12、腰 3）**，双夹爪宽度额外 2 个标量；不能把 32-D 写作含夹爪的最终网络输出，VLA 实际是 **34-D**。
- 遥操作接口依次 **21-D decoupled**、**24-D VR 3-point**、**32-D joint-based**；三任务对照（每组合 40 条示范）只有 joint-based 覆盖全部任务：Cola Placement **86.7%**、Shelf Cup Transfer **80.0%**、Bottle Disposal **85.0%**（均为 progress）。VR 3-point 不能做后两项；decoupled 不能踩脚踏板。
- 同属全身表示，**32-D 机器人关节空间**对比 **81-D SMPL** 在四任务平均进度 **88% vs 75%**；由在线 GMR 重定向降低 SMPL 冗余。
- 前视时延 $Δt$ 在 {0、0.2、0.4、0.6}s 单任务消融中，**0.2s** 为折中：0s 步态更抖；0.6s 进度降为 **13%**；0.2s 为 **67%**。该 67% 来自单任务训练，与四任务混训下 Cola 的 86.7% **口径不同**。

### §3.2、Appendix C.2/E.2：VLA 结构与模型选择

- π₀.₅ 预训练投影最多 32 维，扩到 **34-D** 时使用 weight surgery：保留前 32 维预训练投影，仅初始化 2 个新增槽位；沿用双臂/夹爪旧排序，将腿/腰/根姿态追加；输出绝对关节目标，并输入本体状态。
- 四种接口适配单独消融影响相对小；但原文明示「**去本体 + 相对动作**」联合更改会发生 OOD 漂移及严重失败，不应推导成任意组合都安全。
- 四任务平均：π₀.₅ **91.3%**，PaliGemma **59.6%**，随机初始化 **41.7%**；**验证 action MSE 不能可靠预测实机**。
- 十步 flow matching 约 **90ms**，单步约 **60ms**（作者 RTX 5080 测得），但单步 flow **72.9%**、drifting **70.8%**，低于十步基线 **91.3%**；后两者虽有更低验证 MSE，也没有带来更高任务进度。
- 全身遥操作默认 **每任务 40 条**，Appendix E.2 的 5/10/20/40 条缩放在 20→40 时增益递减；在 8 个训练任务上平均进度 **89%**。

### §3.3、Appendix C.3/E.3/E.4：异构数据的两种不同口径

- **全部 4 个 held-out（Tasks 9–12）**：仅 8 任务全身遥操作模型 **33%**，加 4 个 stationary teleop 任务后 **87%**，全全身遥操作 oracle **94%**（详见摘要及 Figure 5/12；任务 12 为 Pouring）。
- **其中 3 个动作可复用 held-out（Tasks 9–11）**：8-task baseline **36%**、加 HuMI **84%**、加 stationary **89%**、11-task oracle **96%**（官网 Figure/交互口径）；**不得将 84/89/96 直接当成全部 Tasks 9–12 的结果**。
- HuMI **不能有效学会 Task 12 的新倾倒动作**；stationary 同机数据可以。研究者分析原因为 HuMI 摄像头/夹爪域差及 IK 动作分布差，当前 HuMI 更擅长迁移新语义，而非全新运动技能。
- 对每个 held-out 任务的 40 条示范：full loco-manip 平均 **21min**，stationary **13min**，HuMI **7min**（未计全身示范中的过热和重置时间）。HuMI 9–11 从 5、10、20、40 条示范对应 **42%、67%、76%、84%**；40 条效果不能无条件推广至新的运动模式。
- HuMI 处理使用 GoPro 去畸变 + HTC Vive 姿态 + 离线 IK，使数据复用 34-D 状态动作编码；为匹配低层前视策略将 HuMI 动作前移 **0.2s**。

### §4–§6、Appendix D：长程系统对照与限制

- 长程任务从低桌拿第一种水果、中桌拿第二种、再放高架容器；5 种水果按顺序取两种，任务空间 **20 个有序组合**；对每种方法的最终评估**只选取其中 10 组各运行 1 次**（并非 20 个组合全部执行评测）。
- OpenHLM **36 条 full teleop（6 种原水果组合）+ 84 条 HuMI（14 种含 held-out 水果组合）**，合计 **1.14h**，平均 task progress **87.5% ± 3.7%**；GR00T N1.6 **57.5% ± 4.6%**、Ψ₀ **48.8% ± 4.4%**（两者均 **2.70h**），full teleop oracle **97.5% ± 1.7% @ 2.73h**。
- **公平性边界：** 原文 Appendix D.2 将 GR00T 的手部输出替换为双夹爪并扩为三摄像头；将 Ψ₀ 的 36-D 输出映射到同一 decoupled 控制接口并屏蔽空槽。所以这些结果是论文**统一硬件/数据协议下经适配的系统级对照**，不是直接使用两款官方原生配置的全场景结论。
- 论文承认 **仅 G1 一种机器人**、场景与采样规模有限；HuMI 的运动域差尚未突破；本体与动作预测的细节消融缺少机理解释，尤其 val MSE 和实机表现不一致。户外 20 条示范是项目展示，不应写为统一 benchmark。

### Appendix B/C.4：硬件、运行与开销

- G1：双 ChangingTek CTAG2F90-D 夹爪、两个腕部 RealSense D405、头部 Unitree SV1-25；遥操作使用 PICO4U 头显/手柄/腿部 tracker；HuMI 用双 GoPro 夹持器与**5 个 HTC VIVE Tracker**（双手、骨盆、双脚）。
- 默认训练 AdamW、peak lr 1e-4、30k steps、global batch 128、action horizon 50、图像 224×224；作者在 **4×A800 GPU** 上约 **24h**。推理以 **30Hz** 输出 action chunk，**每块 50 步**，最多执行 25 步后再重推（约 0.83 秒）；不是 VLA 每 30Hz 都重新运行一次。
- 已核对官方 README 的训练入口 `src/openpi4OpenHLM/scripts/train_pytorch.py`、服务端 `scripts/serve_policy.py --env SONICG1 --num-steps 10`、机器人侧 `GR00T-WholeBodyControl4OpenHLM/scripts/deploy_stream.sh` 与 `scripts/openpi-eval/main.py` 命令。明确 **未训练 / 未在 G1 上运行代码**。

## 对 wiki 的映射（汇总）

- 主实体：[`wiki/entities/paper-loco-manip-161-154-openhlm.md`](../../wiki/entities/paper-loco-manip-161-154-openhlm.md)
- 站点 / 仓：[`sources/sites/openhlm-project-github-io.md`](../sites/openhlm-project-github-io.md)、[`sources/repos/openhlm.md`](../repos/openhlm.md)
- 161 策展索引：[`loco_manip_161_survey_154_openhlm.md`](./loco_manip_161_survey_154_openhlm.md)
