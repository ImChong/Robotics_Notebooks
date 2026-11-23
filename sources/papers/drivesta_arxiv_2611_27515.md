# DriveSTA：Beyond a Single Human Demonstration: Learning from Scored Trajectory Alternatives for End-to-End Driving

- **类型：** paper / end-to-end autonomous driving / multimodal trajectory planning
- **论文：** <https://arxiv.org/abs/2611.27515>；[arXiv v1 PDF](https://arxiv.org/pdf/2611.27515v1)；2026-11-20 提交，21 页、9 图（含补充材料），未经同行评审的预印本
- **作者：** Xinyao Rao¹、Zecheng Tang²、Liuxuan Sun¹、Zheyuan Lai³、Minjun Yu³、Xun Cao¹、Wei Li¹ ³（通讯）
- **机构：** ¹南京大学（Nanjing University），²北京大学（Peking University），³EABOT.AI
- **项目主页：** <https://rtwotwo.github.io/DriveSTA/>
- **官方源码：** <https://github.com/Rtwotwo/DriveSTA>
- **模型与轨迹资产：** <https://huggingface.co/Rtwotwo/DriveSTA>
- **入库与全文复核：** 2026-11-23
- **核对范围：** 获取 arXiv:2611.27515v1 官方 21 页 PDF，以 `pdftotext -layout` 提取并通读第 1–9 页摘要/引言/相关工作/§3 方法/§4 结果/§5 局限/§6 结论，第 9–11 页参考文献，第 12–16 页补充材料 Appendix A（数据和评测公式）、B（搜寻筛选/模型/训练/JEPA/消融/耗时），第 17–21 页 Appendix C 图 S1–S5 的图注；图像本身仅按图注、正文及官方项目页作语义核查，未逐像素复核细节。同步静态核查官方 README / MINING / INFERENCE / 模型源文件。**未独立复现实验**。
- **通读：** 已完成

## 一手原文核心摘录（以论文 §/Table/Eq 为锚）

1. **问题界定（Abstract / §1）：** 单条记录轨迹不能提供同一场景多种可行行为的几何与质量监督；ChainFlow-VLA 的 VLM 条件精修推理开销高。本文在保留 AR Proposal + residual diffusion 表达下，让离线多备选监督生成、精修和评分，并通过 JEPA 提供共享时序特征。
2. **轨迹构造（§3.2 Eq.1 / Appendix B.1）：** 使用五次 smoothstep `b(r)=6r^5−15r^4+10r^3`，`x_t(u)=x_t^expert+b((t−1)/(H−1))u`，在记录轨迹上做自车系纵横平滑终点偏移；初始纵向偏移 [−5,8] m、横向 [−3,3] m，可恢复扩展到 [−15,40] / [−12,12] m；按固定时间采样及轨迹切线重建方向，跟踪后与日志交通流比较并由 NAVSIM v1 计算评分。
3. **搜索与筛选（§3.2 / Appendix B.1）：** 每批 64 查询，25% 全局探索、其余在优质候选附近搜索；预算 3,072→8,192→32,768；保留跨质量区间 3,072 路径（训练 scorer），另要求 256 条 `PDMS≥95` 的有效候选入池。筛选须通过 NC/DAC/TTC/comfort/DDC、车体足迹、轨迹连续、跟踪误差和方向检查；多样性要求候选两两全路径 ADE≥0.25 m，至多保留 32 条高分目标。**来源统计：** 95,998 有筛选目标的真实场景、2.61M 目标，平均约 27 条；7,265 场景没有通过筛选的目标仍保留专家和评分监督。
4. **Stage 1（§3.3 Eq.2/3）：** 四个相机视角的 DINOv2+LoRA 生成 64 个 AR 自行车积分轨迹；对轮换至多 8 个筛选目标做一对一 Hungarian 匹配，目标几何距离为 `|dx|+|dy|+0.25|dheading|` 平均，`L_G=L_E+0.25 min(1,e/2)L_A`；score head 预测六个组件，使用 BCE + 0.5×聚合评分平方误差，并用 64 条跨质量重播轨迹辅助标定。
5. **Stage 2（§3.4 Eq.4/5 / Appendix B.4）：** 每条 detached Proposal 最多选择 4 个兼容目标，门槛分别为全路径位置 ADE≤3 m、终点≤6 m、航向误差≤0.5 rad；把目标与 parent 的位置/角度差作为 residual，以 DiT v-prediction+几何损失学习，重建后选择误差最小目标；未匹配轨迹用较小权重 0.05 的零残差监督。推理是 12 次确定性 DDIM 迭代精修 64 条候选，而非仅精修已选中的一条。
6. **共享 JEPA（§3.4 Eq.6 / Appendix B.3/B.4）：** 从 Drive-JEPA checkpoint 初始化 ViT-L；训练阶段用 8 帧视频掩码预测 EMA 目标特征、`L_J` 为 L1 特征损失。规划分支编码前一帧与当前前视图，输出经 stop-gradient 分别送到 DiT 条件适配器和 Scorer 适配器，**共享一次 observed-frame 编码**。未来帧、predictor、EMA 目标网络仅用于训练；并非在线推理使用未来视频或独立大 VLM。
7. **评分与选择（§3.3–3.5 Eq.3/7/8）：** Stage 2 对全部精修结果重新做 evaluator 标签及 scorer 学习，防止沿用精修前的候选排序；Rank 用 0.005 质量差筛正样本并做 pairwise logistic，v1 真实数据 `λ_R=0`，混合 `λ_R=0.05`；推理取 64 条路径预测 aggregate score 最大者。NAVSIM v2 实验采用单独选择的 S1 + Rank checkpoint，不能将其与 v1 S2 混称。
8. **模型/训练（Appendix B.2 Table S2/S3）：** 4 个当前相机视角 1148×672；DINOv2 ViT-S/14 LoRA rank32；64 个场景 token×256、一个 ego token；JEPA ViT-L width 1024；DiT 12 blocks/8 heads/width256；64 proposals×8 个 `(x,y,θ)` 位姿（4 s/0.5 s）；训练 Stage1 25 epoch / Stage2 40 epoch，8×H20 GPU、每 GPU batch6、AdamW、BF16（评测 FP32）；**一套配置一个训练随机种子**，每次报告该 run 完整评测中的最好 checkpoint。
9. **NAVSIM v1（§4.2 Table1）：** 12,146 NAVTEST 场景，DriveSTA S1 94.76、S2 95.14 PDMS（约 103k 训练）；DriveSTA-Scale S1 95.09、S2 **95.55**（232,672 真实+合成场景）。S2 在约 103k regime 中居表列模型之首；Scale 结论限定于论文比较的数据规模和方法。项目页另列 FPD-Drive-Pro 95.69，应特别核查是论文引用/发布版本和精度口径，不能混为同一训练预算。
10. **闭环 HUGSIM（§4.2 Table2 / Appendix A.3）：** 345 场景（49 E、126 M、86 H、84 X），DriveSTA S2 的 RC/HD-Score 为 55.2/44.0，Scale S2 **56.1/46.7**；在 Extreme 子集 S2 相对 S1 更差，Scale comfort 从 99.6 降到 95.0，存在性能权衡。评测每 0.25s 重规划、common controller、50ms 控制预算，不等于网络 forward 延迟低于 50ms。
11. **匹配时延（§4.2 Table3 / Appendix B.7）：** 相同 12,146 Navtest、H20、batch1、64 candidates、S2 12 DDIM steps，ChainFlow-VLA **1294.68±4.47 ms**，DriveSTA S2 **349.30±5.21 ms**（约 −73%）；S1 **148.99±0.85 ms**。计时从 GPU 已驻留输入到轨迹选择，**排除磁盘/预处理/传输/仿真打分**，不是车端 end-to-end wall-clock。ChainFlow-VLA 使用 BF16 InternVL、eager attention 且未启用 FlashAttention2，需保留作为对比边界。
12. **消融（§4.3 Table4–6）：** Stage1 dense matches 在控制变量下将 selected PDMS 93.92→94.53、Oracle 99.31→99.77；加入 Replay、Rank 并非对真实域都正收益。Stage2 Screen+多目标兼容匹配+JEPA adaptation 分别改善所选指标，视频条件化 Fixed-pool 对 S2 scorer 的初始/精修候选分别提升 4.41/3.20 PDMS（配对区间见 Table6），但 scorer 的价值受候选分布和阶段约束。
13. **NAVSIM v2（§4.4 Table7/8）：** DriveSTA-Scale S1 在 Navtest EPDMS **91.32**，低于 WA-JEPA **91.7**；在 Navhard **56.13**，低于 DriveZero-Scale **57.1**；在 Navhard 合成反应延续集的 extended comfort 由 70.6→32.3，表明 v1 单条轨迹优化不保证连续规划平滑。
14. **局限（§5 / Appendix B.3/B.5/B.6）：** 参考轨迹平滑形变限制离散驾驶动作覆盖；评分和有效性依赖非反应式 evaluator；真实交通参与者不会因备选自车路径改变行为；逐候选监督未显式约束跨重规划周期的舒适一致性。JEPA 初始化 checkpoint 的完整 driving-domain pretraining 配方尚未公开；每配置一个训练种子及挑选最好完整 checkpoint，不能推断独立重复稳定性。
15. **视觉与补充材料（Appendix C 图 S1–S5）：** 图注与正文显示 JEPA cross-attention、精选 NAVSIM 碰撞对照及 HUGSIM rollout；可视化样本来自 41 场景筛选池，注意力权重**不是因果解释**、精选案例不代表统计失败率。图像细节未逐像素复核；定量以原文 Table1–8 / S1–S4 为准。

## 评测口径与工程结论

| 指标 | 官方原文结果 | 必要限定 |
|---|---|---|
| NAVSIM v1 Navtest PDMS | 95.1424（Base S2）/95.5520（Scale S2） | 12,146 场景；不与 EPDMS 等价 |
| HUGSIM 闭环 | 56.1 RC /46.7 HD-Score（Scale S2） | 345 episodes，加权难度分布；Extreme 退化 |
| NAVSIM v2 EPDMS | Navtest 91.32、Navhard 56.13（Scale S1） | 不是 S2 结果；低于所比较的个别基线 |
| Planner forward latency | 349.30 ms 对 1294.68 ms（−73%） | H20 batch1/64 candidates/12 DDIM；排除 I/O 与预处理 |

## 对 wiki 的映射

- [DriveSTA 论文独立实体](../../wiki/entities/paper-drivesta.md)
- [官方主页归档](../sites/drivesta.md)
- [官方源码归档](../repos/drivesta.md)

**提醒：** 本归档是官方 v1 PDF 全文方法、实验和附录的事实编译；没有复现模型运行和原始训练，也没有将注意力热图作为可解释性因果证明。
