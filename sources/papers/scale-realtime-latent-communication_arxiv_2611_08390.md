# SCALE: Real-Time Latent Communication for Cyber-Physical Agents（arXiv:2611.08390）

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-scale-realtime-latent-communication.md`）

- **标题：** SCALE: Real-Time Latent Communication for Cyber-Physical Agents
- **作者 / 机构（以论文首页为准）：** Shanghao Shi、Jian Xiao、Ching-Hsiang Chan、Sanjoy Baruah、Yevgeniy Vorobeychik、William Yeoh、Ning Zhang（Washington University in St. Louis）；Sastry Kompella（Nexcepta Inc.）
- **类型：** paper / 通信 / 多智能体协同
- **arXiv：** <https://arxiv.org/abs/2611.08390>（v1，cs.RO，2026-11-05）
- **PDF：** <https://arxiv.org/pdf/2611.08390v1>（2026-11-09 取得，pdftotext 转文本，另渲染第 4 页核对图 2）
- **代码：** <https://anonymous.4open.science/r/SCALE-4E80/>（匿名评审仓库，README 经 anonymous.4open.science 文件接口静态读取，未运行）
- **入库日期：** 2026-11-09
- **一句话说明：** 利用物理世界的时空连续性，把当前隐表示 token 与接收端已确认的隐状态做相似度匹配来复用冗余信息，再按「语义收益 / 通信代价」和截止时间用贪心 + EDF 可行性检查决定发哪些消息、每条发多少 token。
- **核对范围：** 通读 v1 正文 §1–§6（摘要、引言、相关工作、问题建模、方法、评测、结论、局限）、表 1–4、公式 (1)–(7)、图 1–4 的图注与图 2 页面渲染；附录 A（调度算法 1）、附录 B（数据集与设置，表 5）、附录 C（运行开销表 6、示例图 5–7 的图注、AI 使用声明）。图 3 以及附录图 8–10（时延 CDF）只读图注与正文转述，未逐图核对曲线；参考文献只读与比较对象相关的条目，未读被引论文本身。代码仓库只读 README 摘要，未读源码细节、未运行。
- **通读：** 已完成

## 核心摘录（均出自原文）

1. **问题（§1、§3）：** 现有隐空间通信（LLM 智能体之间交换 hidden state / KV cache）几乎不考虑带宽与任务截止时间；协同感知方法考虑带宽和时延，但面向单一感知任务。论文建模为：多个任务产生的隐消息共享一条带宽受限且可时变的链路，采用 firm deadline 模型（消息只有在 `C_i ≤ d_i` 时才有贡献），目标是在截止时间前最大化送达的总语义收益 `Σ U_i(K_i)·1[C_i ≤ d_i]`（式 2）。
2. **方法三阶段（§4）：** (1) 语义变化评估：对每个当前 token，在接收端已确认（acknowledged）的隐状态里找余弦相似度最大的 token（式 3），变化分数 `s = 1 − c`（式 4），相似度 `c ≥ τ` 才可复用；(2) 效用–代价画像：按变化分数降序排序，`U_i(K)` 为前 K 个分数之和，`B_i(K) = K·b_tok + B_meta(K)`（式 5，元数据含 token 索引、对应关系、参考状态版本）；(3) 实时调度：每次给某条消息多加一个 token，取边际收益/代价 `ρ` 最大且 EDF 顺序下所有已选消息仍满足 `C ≤ d` 的增量（式 6、7，算法 1），直到无可行增量，再按 EDF 发送。接收端用收到的 token 加上复用的旧 token 更新状态并回 ACK。
3. **适用范围（§1）：** 作用于现有模态编码器产出的隐 token，不改底层模型；论文称支持不同感知模态与下游任务。
4. **评测设置（§5.1、附录 B 表 5）：** CooperScene（真实协同驾驶，3,936 检测任务 + 18,693 预测目标，CoBEVT + 冻结的 MTR）、OPV2V（仿真，528 检测 + 265 分割任务）、LIBERO（SmolVLA，**单个**语言条件操作任务，每种设置 45 个 episode）。检测截止 100 ms、预测/分割 200 ms；LIBERO 截止由机器人剩余缓冲动作决定。带宽取完整隐传输所需平均带宽的若干比例（CooperScene 全量约 118.5 Mbps、OPV2V 约 153.0、LIBERO 约 13.46）以及时变链路。基线：Full-send、Uniform 压缩、CoSDH、V2X-ViT（CooperScene）、CooperTrim、Where2comm（OPV2V），均使用 EDF 发送顺序。
5. **CooperScene 结果（§5.2，表 1）：** 检测 20 Mbps 下 SCALE AP 51.53，高于最强基线 CoSDH 的 31.34（差 20.19），按时率 80.36%（CoSDH 48.02%）；预测在固定带宽下成功率 60.82%–60.85%、按时率不低于 99.82%；时变带宽下检测 AP 56.09 / 按时率 94.52%，预测成功率 60.81% / 按时率 99.71%。Full-send 在固定带宽下检测按时率为 0.00%，但表中 AP 仍为 22.77（论文未解释该 AP 是如何在 0% 按时下得到的）。
6. **OPV2V 结果（§5.2，表 2）：** 分割 mIoU 与按时率在所有设置中最高（15 Mbps：47.14 对 Where2comm 37.17，按时率 97.09% 对 7.82%）；检测 AP 除 8 Mbps 外最高（8 Mbps：83.41，低于 CooperTrim 的 84.25，按时率 37.74%，而 Uniform 为 46.49%，高于 SCALE）。
7. **LIBERO 结果（§5.2，表 3）：** 3.4 Mbps 下 SCALE 成功率 44.44%，Full-send 与 Uniform 均为 0.00%；5.0 Mbps 为 86.67%（Full-send 24.44%，Uniform 0.00%）；10.1 Mbps 为 97.78%；时变带宽 100.00%。成功率定义为 280 步内完成的 episode 比例。Uniform 在 3.4 Mbps 的按时率 74.47% 与 SCALE 的 77.76% 接近，但成功率为 0，论文据此说明「按时送达不等于任务成功」。
8. **实时性（§5.3）：** 按时率分母包含未送达消息；CooperScene 20 Mbps 下检测按时 80.36%（Uniform 68.64%），预测 100.00%（Uniform 94.44%）；检测延迟 CDF 在截止前陡升，作者解释为与贪心分配带宽直至截止边缘一致。
9. **实体蜘蛛机器人案例（§5.4，表 4）：** 树莓派蜘蛛机器人，相机 + 超声波；检测任务通信截止 1,000 ms，超声距离测量作为背景流量；路线为 55.4 秒的微缩城市巡逻。对比 Raw RGB 与 Full-send 隐特征，带宽 1 / 2 / 8 Mbps。1 和 2 Mbps 下两个基线成功率与按时率均为 0，SCALE 成功率 77.62% / 91.67%，按时率 100.00% / 96.43%；8 Mbps 下三者均为 100%。
10. **开销（附录 C，表 6）：** i9-13900K、4 核，CooperScene 上每条消息总开销 0.813–0.884 ms。
11. **作者自述局限（Limitations）：** (a) 通信节省依赖时空冗余，物理世界突变会减少可复用信息、增加维持性能所需通信；(b) 调度用当前链路速率判断可行性，无法应对短于最小调度间隔的突发带宽下降。
12. **论文未给出 / 未说明（本档观察）：** 相似度阈值 τ 的取值正文未找到；真实机器人案例仅一种检测任务且无场景突变对照；LIBERO 仅 1 个任务；基线并非为「截止时间 + 多任务共享链路」设计，只是被统一加上 EDF 顺序；ACK 本身的延迟与丢失在正文中未建模说明；未与 LLM 隐通信方法（KVComm、Cache-to-Cache 等）做实验对比。

## 与外部说法对照（具身智能小站 2026-11-09 条目）

- 小站：「复用连续观测中相似的隐表示，再按语义收益、带宽和截止时间决定消息及 token 数量」——与 §4 一致。
- 小站：「测试覆盖协同感知、操作与实体蜘蛛机器人」——与 §5 一致（CooperScene / OPV2V 为协同感知，LIBERO 为操作，§5.4 为蜘蛛机器人）。需补充：LIBERO 只测单任务，蜘蛛机器人为检测 + 距离测量，不是操作。
- 小站：「场景突变会削弱信息复用带来的收益」——与 Limitations (a) 一致；论文没有做对应的突变场景实验，只是文字承认。
- 小站未写机构：论文首页为 Washington University in St. Louis（另 Nexcepta Inc.）。

## 对 wiki 的映射

- [paper-scale-realtime-latent-communication](../../wiki/entities/paper-scale-realtime-latent-communication.md)
