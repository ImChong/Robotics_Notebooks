# Shape Your Body: Value Gradients for Multi-Embodiment Robot Design

> 来源归档（ingest）

- **标题：** Shape Your Body: Value Gradients for Multi-Embodiment Robot Design
- **类型：** paper（机器人共设计 / 多具身 RL / 价值函数梯度搜索）
- **状态：** arXiv v1（cs.RO，2026-05-30）；项目页与 PDF 标注 Under Review
- **原始链接：**
  - arXiv abs：<https://arxiv.org/abs/2606.00702>
  - arXiv PDF（本次通读版本 v1）：<https://arxiv.org/pdf/2606.00702v1>
  - arXiv HTML：<https://arxiv.org/html/2606.00702v1>
  - 项目页：<https://nico-bohlinger.github.io/shape-your-body/>
  - PDF（TU Darmstadt）：<https://www.ias.informatik.tu-darmstadt.de/uploads/Team/NicoBohlinger/shape_your_body.pdf>
- **作者：** Nico Bohlinger, Jan Peters
- **机构：** Technical University of Darmstadt；Robotics Institute Germany (RIG) / DFKI / hessian.AI
- **入库日期：** 2026-06-02（2026-11-12 按全文通读重写核对）
- **核对范围：** 下载 arXiv v1 PDF 全文（27 页）并逐段读完正文 §1–§6 与附录 A（环境 / 奖励 Table 1 / 50 机 Table 2 / PPO Table 3）、B（critic 结构与消融 Table 4–5）、C（优化器超参）、D（λ 敏感性 Table 6）、E（跟踪误差）、F（RL 共设计基线适配）、G（50 机结果）、H（设计分析与 Fig 11–12 的文字部分）、I、J（负面结果）；柱状图 / 热力图 / 轨迹图（Fig 3、5、8–11、13–15）只读到图注与正文引用的结论，未逐点读取图中数值；参考文献表只扫读。论文未提供代码（项目页标注代码「soon」），故无源码可核。
- **通读：** 已完成
- **一句话说明：** 先在最多 **50** 台腿足机器人上用 PPO 训练 URMA 多具身策略与 direct-design critic，再冻结并用 **Value-Gradient Design Search (VGDS)** 在软信赖域内沿价值梯度优化机体与 PD 等连续参数；单机器人设置下初始训练约 7–9 h，之后每个新设计约 1–2 min 搜索。

## 核心论文摘录（按论文章节核对）

### 1) 问题与动机（§1、§2）

- 经典共设计是双层优化：外环提设计、内环为每个候选训练或适配控制器（§1）。近期 RL 共设计（FEACRL、Transform2Act、BodyGen、Stackelberg PPO）在单次训练内共享经验，但「每个新机器人、设计空间或任务都要把整条共设计环重跑」（§1）。
- 作者主张：先训一个多具身策略 + 价值函数，冻结后当可复用设计模型，下游设计搜索便宜且可对多个候选批量并行（§1）。
- 与 FEACRL 的区别：FEACRL 用 SAC 的 Q 函数配 PSO 给候选设计打分，但仍绑定到单机器人的共设计运行（§2）。

### 2) 设计空间与训练（§3、§3.1、附录 A）

- 归一化设计向量 f ∈ [-1,1]^{d_design}，可微映射 e = Φ(f) 到物理参数（关节原点与范围、执行器力 / 速度限、阻尼、刚度、armature、摩擦、PD 增益、质量、惯量、质心偏移、连杆几何、足参数）；Φ 内可加硬物理边界；本文未加设计专属奖励项（§3）。
- 每个并行环境在 episode 重置时采样 f_i ~ U([-c_t, c_t]^d)，episode 内固定；性能课程系数 c_t 扩到 1 覆盖全空间；PPO 最大化 E_f[J(θ, Φ(f))]（§3.1，式 2）。评估始终用完整设计空间（A.3）。
- URMA：每关节观测 o_j + 描述向量 d_j，attention 聚合；原 critic 中 d_j 只进 attention 的 key。**direct-design critic**：编码器 g_ψ(o_j, d_j) 同时吃观测与描述；K 个 value head 共享逐关节 / 逐足编码器，取均值（§3.1）。作者称这「在经验上带来更强的设计梯度」。
- 附录 B Table 4（早期 Go2 设计任务，10 个初始设计均值）：原 URMA critic 1 头 latent 4 为 +10.39，3 头 latent 4 为 +13.57；direct-design critic 3 头 latent 4 仅 +6.41，latent 32 最好 +17.76，latent 64 为 +14.35。结论：latent=4 时 direct-design 反而更差，需足够 latent 容量；最终取 K=3、value latent 32。critic 共 5,378,312 参数、策略 945,430（Table 5）。

### 3) VGDS（§3.2、附录 C、J）

- 冻结策略与 critic；从全设计空间 rollout 收集状态库 S；目标 Ĵ_λ(f) = mean_m V̄(s_m, Φ(f)) − λ‖f − f_ref‖²/d_design（式 5）。作者在初步实验中观察到：无约束最大化常把设计推向边界，critic 预测高回报但真实性能崩溃（§3.2）。
- 更新（式 7）：minibatch 上梯度 → Adam 步 → 每维更新 clip 到 δ_max → clip 到 [-1,1]。附录 C：Adam 50 次迭代，学习率 0.05，δ_max = 0.1，minibatch 64 个状态，λ=100；最终设计用冻结策略 rollout 200 个 episode 取平均回报。
- 对比搜索法（BO、CMA-ES、PSO、CEM、DE、ARS、TuRBO、GC-PFO、随机搜索）全部用同一个冻结 critic 与同一目标 Ĵ_λ；超参在「早期版本的扰动起点实验」上调过，并令各法迭代数 / critic 评估数量级相近（附录 C）。
- 附录 J 负面结果：硬信赖域投影与软 L2 相当但半径更难调；基于 head 分歧的不确定性信赖域不可靠（小集成仍可能一致地做出灾难性外推）；多起点、刷新状态库、噪声 / 调度等无稳定收益；「更多迭代有时因过度优化 critic 反而变差」；有用的设计信号「主要是稠密高维的」。

### 4) 实验（§4、附录 A、D–H）

- 环境（A.1）：RL-X + MJX，仿真 200 Hz / 控制 50 Hz，episode 最长 20 s（1000 步）；速度跟踪 locomotion；奖励 Table 1 含 23 项；动作 q_target = q_nom + σ_robot·a。
- 50 台基座机：15 四足、31 双足 / 人形、4 六足；d_design 从 190（Bolt、Tron 1）到 1177（STAR1）（Table 2）。
- **单机器人（§4.1）**：Go2（358 维）、MIT Humanoid（514）、Golem 六足（688）；各自用 4096 个并行环境训 1.0×10^9 步（A.5）；每机 10 个均匀采样的 f_init。Fig 3：VGDS 与其他搜索法都显著改进 f_init，接近或超过名义 f_ref；Go2 和 Humanoid 上 VGDS 与最强基线持平，Golem 上改进最大；GC-PFO 是最强基线（唯一另一个用梯度的方法）。
- **λ 敏感性（附录 D Table 6）**：λ=0 时 Go2 的 ΔR 为 −18.93（比随机初始设计还差），Humanoid +2.21、Golem +0.50；λ=100 时 Go2 +20.60、Humanoid +28.84、Golem +16.85；Humanoid、Golem 在更大 λ 略高（Humanoid 31.20 @200、Golem 17.27 @300），作者承认「各机分别调 λ 可取得更好结果」。
- **对比 RL 共设计（§4.1，Fig 4，附录 F）**：Schaff2019、FEACRL、Transform2Act、BodyGen、Stackelberg PPO 均被「大幅改造」成 URMA + PPO 的连续设计版本；作者明说这「不完全复现原方法」，结果应理解为底层思想的比较。VGDS 与它们「持平或略好」；RL 基线耗时随初始设计数线性增长；首次训练约 7–9 h，之后每个设计约 1–2 min，批量 critic 推理可进一步缩短。
- **泛化（§4.2，Fig 5）**：留出目标机的 morphology-class 训练集（7680 环境、3.7×10^9 步）vs 含目标机的 50 机训练集（15040 环境、3.7×10^9 步，单张 RTX 6000 Ada 约 2.5 天，A.5）。全 50 机训练下 Go2 和 Humanoid 的 VGDS 设计可超过 f_ref；Golem 上仅六足子集（只有 4 台）训练更好，作者归因为「很可能」是全集被四足 / 双足主导。附录 G：50 机全部做设计，部分大型人形「似乎未完全收敛」，改进较小。
- **设计分析（§4.3，附录 H，Fig 11–12）**：Humanoid 最强变化是名义关节位置与增益、足缩小；仅把优化后的增益拷进初始设计，回报由 5.8 升到 12.5。Golem：动作缩放降低 12.2%、P 增益降 4.1%、D 增益升 4.4%；Fig 12 文字：只改 PD 增益会变差，降低 action scale 有帮助。Go2：后腿关节轴、足几何、前髋 / 小腿速度限。没有单一参数组能解释全部提升。
- **健康检查（附录 H）**：三个单机器人实验中没有任何优化维度超过 |f_i|>0.9；每维平均归一化变化 0.027–0.042；17 个部分修改设计上 critic 预测与真实 rollout 回报的 Spearman ρ=0.40（Humanoid 0.50、Go2 0.60、Golem 0.83）。
- **跟踪误差（附录 E）**：回报提升不一定意味着 x-y 跟踪误差降低；单机器人设置中 VGDS 与 GC-PFO 的误差下降几乎相同。
- 附录 I：优化轨迹显示不同初始设计收敛到相似的机器人专属最优，作者称之为「多样性坍缩」。

### 5) 局限（§5）与结论（§6）

- 仅固定拓扑上的连续参数，不能增减关节 / 连杆（URMA 没有拓扑梯度）；依赖策略与价值函数的覆盖和精度，信赖域能缓解外推，但「不总有足够好的参考设计」；全部实验为较简单的 MuJoCo 仿真，未评估可制造性或真机可迁移性（§5）。
- 展望：拓扑变化、更复杂设计约束、任务专属设计目标、潜空间信赖域（§6）。
- 致谢：波兰 NCN Weave UMO-2021/43/I/ST6/02711；DFG PE 2315/17-1。

## 相关资料（交叉核对，非正文逐字摘录）

| 资料 | 链接 | 备注 |
|------|------|------|
| 交互演示站 | <https://nico-bohlinger.github.io/shape-your-body/> | 浏览器内 URMA 策略 + Reference / Co-Design 滑条 |
| 项目站归档 | [shape-your-body-nico-bohlinger.md](../sites/shape-your-body-nico-bohlinger.md) | TL;DR、公式、50 机训练集、Citation |
| URMA 原始 | PMLR [bohlinger25a](https://proceedings.mlr.press/v270/bohlinger25a.html) | 论文引用 [5]（CoRL 2024） |
| URMA 规模化 | arXiv:2509.02815 | 论文引用 [42]，课程式极端 embodiment 随机化 |
| 代码 | 项目页标注 *Code (soon)* | 未见具体仓库地址 |

## 当前提炼状态

- [x] arXiv v1 PDF 全文（正文 + 附录 A–J）通读，核对范围见上
- [x] 项目页交互说明与训练集规模已对齐
- [x] wiki 映射：`wiki/entities/paper-shape-your-body-value-gradient-design.md`
