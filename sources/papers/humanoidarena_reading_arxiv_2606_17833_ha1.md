# HumanoidArena 论文通读（分层全身基准视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-humanoidarena.md`）

- **标题：** HumanoidArena: Benchmarking Egocentric Hierarchical Whole-body Learning
- **作者：** Taowen Wang, Zikang Xie, Bin Yang 等（HKUST-GZ、北京工业大学、HIT 深圳、深圳北理莫斯科大学、京东探索研究院）
- **类型：** paper / benchmark / humanoid / hierarchical-control
- **原文：** <https://arxiv.org/abs/2606.17833>（2026-11 通过 arXiv PDF v1 获取全文，15 页）
- **一句话说明：** 把「高层策略 -> 40D 中间全身动作 -> 低层 GMT」的接口本身作为评测对象，在 7 项腿关键 HOI/HSI 任务、三类扰动与跨 GMT 部署下诊断策略与跟踪器的兼容性。
- **核对范围：** 通读正文 §1–§5（引言、相关工作、§3 基准设计、§4 实验与 Finding 1/2、结论）及附录 A–J（任务与成功判据、数据管线、40D/64D 接口与 GMT 适配器、评测与扰动配置、t-SNE 分析、失败分类与 Football 案例、复现与许可、表 S7/S8、局限）；表 S6 超参仅摘录动作块长度、学习率、批大小等主要项；参考文献与逐格附表数值未逐项核对。未阅读 TWIST2 / SONIC 原文，其细节仅依赖 wiki 既有页面。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 既有基准要么评端到端控制栈或固定动作表示，要么偏重运动或操作，缺少对 policy–tracker 接口的受控评测；贡献为分层协议、腿关键任务与数据管线、扰动与 GMT 条件评测协议、基线与发现。
- **协议（§3.2，式 1–2）：** 高层 u_t ~ π(·|I_ego, p_t, ℓ)，p_t 为 64D 规范本体状态（根朝向 6D、29 关节位置与速度），u_t 为 40D 动作（根水平移动、根高度、根朝向、29D 关节目标、双手开合）；各 GMT 经适配器 ψ_m 映射后输出关节目标 q_{t+1}=G(m)(ψ_m(u_t))。不用 SMPL 动作表示，以保持动作空间可处理。
- **基线（§3.2、§4.1）：** ACT、Diffusion Policy（DP）、Flow Matching（FM）、π0.5；同一观测、动作接口与 GMT，各训练 100k 步；表 S6 中动作块长度均为 20，批大小 64，学习率 ACT 1e-5、DP 1e-4、FM 2e-5、π0.5 2.5e-5。
- **数据（§3.3、附录 B）：** Isaac Lab 内 VR 第一人称遥操作，经 GMR 得 35D 参考；TWIST2 用 mimic 风格跟踪指令，SONIC 用机器人运动编码器模式；每任务每 GMT 100 条成功示范，共 1400 条；50Hz；基准结果用前置 ego 相机，仅成功示范入训练集。
- **任务（§3.4、附录 A）：** HOI 为 Football、DoubleDesk、P&PBox；HSI 为 OpenDoor、SitSofa、Boxing、VisNavi。跌倒判据：躯干倾斜 60° 且连续 5 步确认；75° 或 50N 接触力加倾斜即终止；站立判据根高 >= 0.45m。
- **评测（§3.5、§4.1、附录 D）：** 每配置 3 种子 x 20 次 = 60 回合；指标 SR、平均跌倒率 AFR、跨 GMT 绝对下降 Δcross 与保留率 Ret。视觉扰动 = DomeLight 换为随机朝向与强度的 DistantLight；语义扰动 = 保持目标与功能不变的资产替换或干扰物；执行扰动 = 扩大物体初始化范围（如 Football 球范围由 ±1.2 增至 ±1.45）。
- **In-GMT 结果（表 1、Finding 1）：** TWIST2 下 FM 的 HOI/HSI 均值最佳（36.11% / 58.75%）；SONIC 下 DP 最佳（52.22% / 65.83%）；π0.5 在 SONIC 的 DoubleDesk 最佳（43.3%），AFR 最低；最佳策略随 tracker 而变，结论是性能属于 policy–tracker 对。
- **扰动结果（图 3）：** 视觉扰动造成最一致的下降，HOI 尤甚（如 TWIST2-HOI 上 ACT 28.33% -> 14.44%、FM 36.11% -> 18.33%）；三类扰动暴露不同失败模式。
- **跨 GMT（表 2、Finding 2）：** 换后端后大幅下降且保留率低；T->S 平均下降约 39.9%，S->T 约 36.0%；S->T 平均 AFR 7.56% 高于 T->S 的 1.36%（DoubleDesk 32.08%）；结论是策略默认并未学到与 tracker 无关的意图，部分特化于示范 tracker 的动力学与纠偏行为。
- **t-SNE（附录 E）：** 固定 DP 策略，对 SONIC 与 TWIST2 观测预测的 40D 动作常形成各自分簇，说明差距也出现在策略输出分布上。
- **失败分类（附录 G）：** 感知定位、接近对齐、接触前姿态、接触执行、接触后恢复、tracker 引起的执行失败；Football 案例显示首次踢球无效后可重新定位并二次击球。
- **合并训练（附录 I，表 S8）：** TWIST2x7 下 DP 的 HSI 由 55.83% 升至 73.33%；SONICx7 下 DP 的 HOI 由 52.22% 降至 29.44%；TWIST2+SONIC x14 的 DP HSI 为 68.12%、HOI 为 31.67%；作者归因于 GMT 条件化的动作分布失配。
- **局限（附录 J）：** 任务与场景覆盖有限，未来可扩展到家居、多物体重排、人机交互、工具使用、室外与长程任务。
- **算力（附录 H）：** 训练 8 卡 H200，评测 16 卡 RTX 4090（需图形渲染）。

## 对 wiki 的映射

- [`wiki/entities/paper-humanoidarena.md`](../../wiki/entities/paper-humanoidarena.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2606.17833>
