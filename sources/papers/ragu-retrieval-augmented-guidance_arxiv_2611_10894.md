# Zero-Shot VLA Adaptation with Retrieved Guidance（arXiv:2611.10894）

> 来源归档（paper）

- **标题：** Zero-Shot VLA Adaptation with Retrieved Guidance（方法名 RAGu，Retrieval-Augmented Guidance）
- **类型：** paper
- **作者 / 机构：** Subin Kim, Anh-Quan Pham, Dinesh Jayaraman；University of Pennsylvania（论文首页；项目页署名为 Anonymous Authors）
- **arXiv：** <https://arxiv.org/abs/2611.10894>（读取版本 v1，2026-11-07，cs.RO）
- **PDF：** <https://arxiv.org/pdf/2611.10894v1>
- **项目页：** <https://raguvla.github.io/>
- **数据集（非代码仓库）：** <https://huggingface.co/datasets/raguvla/real-world-evaluation>
- **代码：** 项目页与论文正文均未给出代码仓库链接（项目页外链只有上述 HF 数据集），未核实到开源代码
- **入库日期：** 2026-11-11
- **一句话说明：** 不改 VLA 权重、不扩上下文：从 DROID 等域不匹配的演示库检索轨迹，经语义对应 warp 到目标场景，再作为能量引导场注入 flow-matching 去噪。

## 核对范围与通读状态

- **核对范围：** 读了 arXiv v1 PDF 全部 8 页正文与附录（摘要、I-VI 节、表 I-IX、图 1-8 及说明、附录的置信区间说明 / 超参 / 失败模式分析 / 人形实现细节、参考文献）；图 3、图 4 渲染成图像核对了柱状数值，其余图表以提取文本与图注为准；项目页静态读了首页文本与链接（视频、交互式 rollout 内容未看）；HF 数据集只读了文件清单（未下载视频或 action_history.json）。未读：被引文献（RICL、Omniguide、VLS、Tether、RoboLab 等）原文；未运行任何代码。
- **通读：** 已完成

## 方法要点（作者自述，论文 III-IV 节）

1. **语义任务过滤：** 用 "pick"、"place" 等关键词在 DROID 语言标注上筛出运动学相近的任务类别（IV 节，图 1a）。
2. **跨域轨迹 warp：** 用 Grounding DINO 在目标场景定位任务相关物体得到目标路点 W'，沿用 Tether 的路点位移插值 warp（III 节公式 α 插值），对应关系按「功能角色」而非物体外观定义（IV 节，图 1b）。
3. **两级检索：** 离线用 DTW 按单条参考轨迹粗筛 top-N；推理时在第一个去噪步用 VLA 预测的动作块经前向运动学得到笛卡尔路径 X_t，选使代价梯度范数 ||∇_{A0} C_i(X_t)||2 最小的候选 i*，之后固定（式 3、4）。
4. **引导：** 引导能量 L_warp = C_{i*}(X_t)，代入 v_guided = v − γ∇L（式 2、5）；权重不更新。
5. **超参（表 IX）：** top-N 真机 8、RoboLab 16、GR-1 3；λ=1.0；γ 真机 0.2-2.0、RoboLab 0.2-1.0、GR-1 0.4/0.5。

## 实验摘录（论文 V 节与附录）

1. **设置：** 真机 10 次 rollout / 任务，仿真 25 次；real-to-real（DROID Franka，5 任务）、real-to-sim（DROID 到 RoboLab，9 任务）、sim-to-real（MetaWorld Sawyer 100 条 bin-picking-v3 轨迹到真机 DROID，Banana-to-Plate 多阶段）、跨形态（DROID 到 RoboCasa 中 Fourier GR-1 人形，5 个桌面任务）。
2. **真机（图 4）：** 平均成功率 π0.5-DROID 20%、Omniguide 40%、RAGu 80%、RICL 0%；Wipe Table 100% 对 Omniguide 30%；Close Microwave 上 π0.5 为 60%，Omniguide 仅 10%，RAGu 90%（柱状图读数）。
3. **RoboLab 3 任务（表 I）：** GR00T-N1.7 1.33%、π0.5 5.33%、RICL 1.33%、VLS 12%、RAGu 25.33%。
4. **RoboLab 9 任务（表 IV / VII）：** π0.5 平均 23.6%，RAGu 1 条轨迹 28.4%、10 条 32.0%、30 条 30.2%、50 条 32.4%、100 条 36.4%（95% Wilson 区间 [30.4, 42.9]）；表 VII 中不同条数下 ToyInBin 等个别任务不升反降。
5. **人形（表 II）：** GR00T-N1.5 w/o FT 平均 33.6% 到 100 条 47.2%（+13.6）；w/ FT 36.8% 到 48.8%（+12）；1 条轨迹时 PlateToPlate 在 w/o FT 下 −12。
6. **sim-to-real（图 6）：** Cabinet→Shelf 上 π0.5 与 Omniguide 均 0%，RAGu 60%（图中另两个迁移方向的数值只读到柱状图，未逐一复核）。
7. **次级检索消融（表 III）：** min grad 与 min dist 路径长度相近（5.67 / 5.79 m），min dist 更慢（7.09 对 8.37 cm/s）且更不平滑（EE SPARC −11.74 对 −10.30）；作者对原因用的是「plausible explanation」，属推测。
8. **开销（表 VI 与正文）：** 作者称每 episode 墙钟时间与无引导基线相差在 6 s 以内（720 步、10 Hz）。
9. **附录失败模式：** RoboLab 900 次 warp 尝试中 65.1% 成功，demo 解析错误 22.0%、场景对应错误 10.4%（图 8），文中另称 34.9% 失败；这些百分比描述 warp 结果而非任务成功率。
10. **π0.5 复测（表 VIII）：** 在 RoboLab 全 45 任务上自测 π0.5-DROID 14.0%，RoboLab 原文报 28.0%，作者归因于原文仅 10 次 rollout 噪声大。

## 论文自述局限（VI 节）

- 假设源库含与目标任务类别大致匹配的演示；对相机位置与遮挡不具不变性（语义对应失败会破坏 3D warp 与引导信号）；引导只能在 VLA 先验范围内重塑动作。

## HF 数据集核对（仅文件清单）

- 400 个文件，按 `任务/方法/success|failure/episode_xx/` 组织，每个 episode 含 `video.mp4` 与 `action_history.json`；方法目录为 ours、pi05、ricl、vlm，任务含 close-the-microwave-door、open-upper-left-drawer、hold-the-red-handle-and-open-the-right-cabinet、place-green-towel-on-plate、wipe-table-with-towel。无数据卡。性质是真机评测 rollout 记录，不是训练数据。

## 对 wiki 的映射

- [paper-ragu-retrieval-augmented-guidance](../../wiki/entities/paper-ragu-retrieval-augmented-guidance.md)
