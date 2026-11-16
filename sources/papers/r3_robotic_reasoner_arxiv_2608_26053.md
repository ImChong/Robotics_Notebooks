# R^3: Training Robots to Reason in Natural Language via Reinforcement Learning

> 来源归档（ingest）

- **标题：** R^3: Training Robots to Reason in Natural Language via Reinforcement Learning
- **短名：** R³
- **类型：** paper
- **作者：** Lehong Wu, Yuxiao Qu, Zheyuan Hu, Ivan Zhang, Limin Wei, Zackory Erickson, Aviral Kumar（卡内基梅隆大学）
- **arXiv：** <https://arxiv.org/abs/2608.26053>（v1，2026-08-26，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2608.26053>
- **项目页：** <https://robotic-reasoner.github.io/>
- **代码：** 项目页写明 Coming Soon；论文未给仓库地址
- **入库日期：** 2026-08-28（2026-11-12 通读后改写）
- **索引来源：** [具身智能小站 9 篇盘点](../blogs/wechat_embodied_station_wam_vla_cross_embodiment_9_papers_2026-08-28.md)（<https://mp.weixin.qq.com/s/FNhRO3KOm8k8CkJEqystQQ>）；CoRL 2026 Glen Berseth 阅读清单（P1，第 27 条）
- **一句话说明：** 把自由形式自然语言推理训成冻结低层操作策略可调用的测试时计算接口：先在专家推理轨迹上 mid-train，再用离线动作数据做单步量表 RL。
- **核对范围：** 下载 arXiv v1 PDF（37 页）转文本后逐段通读正文第 1–6 节（含 Table 1–5、Figure 1–6、公式 3.1/3.2）、参考文献、附录 A（任务与奖励、超参 Table 8/9）、附录 B（VQA Table 10、轨迹信号 Table 11/12、评审器验证 Table 13）；附录 C 的轨迹示例仅读了图注与杂货打包部分示例，附录 D 只读了 D.1 提示词，D.2/D.3/D.4 未逐行读；图像本身（Figure 3、10、12–23）只依据图注与文本，未逐图细看；未获得代码与项目页视频。
- **通读：** 已完成

## 开源状态（步骤 2.5）

- **待发布**：项目页按钮为 **Code (Coming Soon)**。截至入库日无 GitHub 训练仓；论文正文未给出仓库地址。训练用到的开源框架：SFT 用 LLaMA-Factory，RL 用 verl（附录 A.6）。
- 2026-11-12 改写时未重新核查项目页，沿用入库日结论。

## 核心摘录（面向 wiki 编译）

### 摘录 1：问题设定与架构（第 3 节，Figure 2）

- 层级结构：高层 VLM π_θ(z_t, u_t | x_t, g) 先生成推理 z_t，再输出短时程指令 u_t；固定的语言条件低层策略 π_lo(a_t | s_t, u_t) 执行一个固定长度动作块。x_t 是历史上下文，g 是文本长时程目标。
- 历史的实现：直接用上一步的完整回复（含推理）作为历史（第 3.1 节）；专家 Gemini 3 Flash 加历史后 pass@1：line 44.9%→51.0%，V 52.3%→57.6%（Table 1）。
- 基座模型：Qwen3.5-4B（第 4 节、附录 A.6）。
- 作者动机：现有 ECoT 等多把结构化推理当训练期辅助监督，推理时产生推理几乎无额外收益（第 1 节，引 Chen et al.）。

### 摘录 2：两阶段训练（第 3.2、3.3 节，Figure 1）

- Stage I mid-training：在专家推理轨迹（含失败轨迹）上做下一 token 预测 SFT（公式 3.1）；Language Table 上每任务 4 条轨迹 × 64 场景 = 256 条（第 4 节）；2 epoch、取最后一个检查点（附录 A.6）。
- Stage II 单步 RL：只用「历史 + 专家指令」（无专家推理）；模型采样（推理，指令），按奖励 R 用 Dr.GRPO 更新（公式 3.2，优势为组内均值中心化）；每任务 128 条成功专家轨迹（第 4 节）。RL 超参：学习率 2e-6、batch 32、每 prompt 12 次 rollout、温度 1.0、clip 0.2/0.3、KL 与熵系数 0（Table 9）。
- 奖励：R = R_acc + R_len。Language Table 的 R_acc 由 Qwen3.5-35B-A3B 评审器按量表打分：1.0 精确语言匹配 / 0.5 副词不匹配 / 0.25 语义匹配 / 0 其他（附录 A.4 式 A.1）；杂货打包用字符串精确匹配 1.0 或 0（式 A.2）；R_len 为对少于 T=80 词回复的对数缩放惩罚，范围 [−1, 0]（式 A.3）。
- 两个 RL 设计：用中期模型每步采样 48 个回复为上一步补全缺失推理（匹配专家指令者才采用，否则只给上一条指令）；过滤重复指令步，避免模型靠重复上一指令拿奖励（第 3.3 节）。
- 评审器验证（附录 B.3 Table 13）：100 对 prompt-response，人类多数票对 Qwen3.5-35B-A3B 一致 90/100，κ=0.837；人类对人类 94.7/100，κ=0.911。

### 摘录 3：Language Table 评测（第 4 节，Table 2/3/4）

- 14 个长时程积木任务，每场景 8 块；6 个 mid 训练任务 T_M、3 个 RL 任务 T_R、5 个 OOD 留出任务 T_O；RL 用 T_M ∪ T_R，T_O 留出（第 4 节）。每任务 64 个留出场景 × 16 次试验，报 95% 置信区间。
- 结论 1：仅 RL（不 mid-train）已显著超过基座；结论 2：mid-training 后 RL 一致优于仅 RL；1/4 mid 数据在 T_R 与 T_O 上已持平或超过完整 R³（第 4.1 节）。
- 相对指令模仿 IL 的 Δ（Table 2，R³ 减 IL）：OOD 五任务 diag_line +14.2±3.7、rect +3.9±1.6、mid +8.7±5.0、iL +9.9±4.6、clear_half +4.8±3.2；训练域上 V +28.3±4.2、iV +18.6±3.9，但 gris −10.3±4.4（IL 58.1 > R³ 47.8），line −1.5±4.2。
- mid 任务上，未训练、无推理的基座得 56.3，高于 R³ 的 51.0（Table 2，我的观察）；论文自述 IL 在该任务掉点、R³ 优于 IL。
- diag_line 上 IL 与 R³ 都比基座差或接近（论文承认的例外，第 4.1 节）。
- 推理确实起作用的三条证据（第 4.2 节）：(A) VQA 诊断中 R³ 感知与指令推断提升但仍远低于 Gemini（Table 10：指令推断 55.3 vs Gemini 85.1），而操作成绩却接近 Gemini，故不能只归因于静态感知；(B) IL 预训练/共训推理数据只带来部分提升，OOD 上仍弱于 R³（Table 3）；(C) 同一检查点截断推理：V 69.2→59.4（trunc@100）→26.6（trunc@50）→20.3（无推理）；gris 47.8→44.1→25.0→22.1（Table 3）。Figure 5 显示在更难、成功率更低的任务上推理更长。
- 与 ECoT（把末端与物体坐标等结构化信息放入推理）对比：加 ECoT 成分略降平均表现（Table 4，第 4.4 节）；事后（post-hoc）标注推理与采集时记录推理相当，作者推测因标注模型与采集模型同为 Gemini（第 4.4 节，作者推测）。
- 行为分析（第 4.3 节、附录 B.2 Table 11/12，30 个验证场景定性统计）：基座回复平均 1079 字符、19/30 符合格式、3 例幻觉物体；mid 后 563 字符、30/30 格式、0 幻觉；RL 后 696 字符。第一人称规划计数 11→15→23。Figure 6：从基座直接 RL 会大幅改写指令分布，mid 之后 RL 为局部修改；罕见指令（Separate、Touch、ArmMoveAbs）RL 后消失。

### 摘录 4：双臂杂货打包（第 5 节，Table 5，附录 A.2/A.3）

- 环境：MuJoCo 仿真，两台 7 自由度 xArm-7 + YCB 物体，装入小/中/大托盘；人类遥操作数据，分段标注指令（第 3.1、5 节；环境与数据来自作者引用的未公开工作，引文为 Anonymous）。
- 低层策略：在该数据上微调 π0.5（第 5 节）；控制频率 60 Hz，每 300 步下一次指令，单集最长 21,600 步（Table 7）。
- 跳过 Stage I（基座已能给出有用推理，且无推理标注），只做 RL（字符串匹配奖励）。
- 12 个留出任务，每任务 5 种子 × 10 次 rollout = 50 回合，共 600 回合（第 5 节）。平均成功率：基座无推理 19.7±1.9，IL 无推理 **38.0±3.0**，基座有推理 23.3±3.2，R³（仅 RL）**47.9±3.3**；平均进度 55.0 / 65.4 / 56.6 / 73.1（Table 5）。
- 逐任务并非一致占优：Task 7 的 IL 成功率 44.0 对 R³ 10.7±15.2；Task 2 两者均为 100；Task 9 的 IL 86.0 对 R³ 84.0（置信区间重叠）（Table 5，我的观察）。

### 摘录 5：论文自述局限与未来工作（第 6 节）

- 两个实验域都是仿真，且低层语言条件策略固定；上真机是未来工作。
- Stage I 仍依赖专家生成的推理；Stage II 依赖 VLM 评审器给语义奖励，优化的是代理目标而非最终任务成功率；多轮在线 RL 留作未来。
- 层级设计可能带来高层意图与低层执行不匹配；联合训练推理与动作是未来方向。
- 事后推理 vs 采集时推理的比较未系统变换采集者与标注者（第 4.4 节）。

## 对 wiki 的映射

- 页面：[paper-r3-robotic-reasoner](../../wiki/entities/paper-r3-robotic-reasoner.md)
- 方法与任务：[VLA](../../wiki/methods/vla.md)、[GRPO](../../wiki/methods/grpo.md)、[manipulation](../../wiki/tasks/manipulation.md)

## 当前提炼状态

- [x] 方法要点与开源核查
- [x] 论文正文与主要附录通读（见「核对范围」）
- [x] wiki 实体与技术地图回链
