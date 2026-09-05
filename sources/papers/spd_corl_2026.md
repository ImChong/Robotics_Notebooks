# Pre-training Visual Dexterity in Simulation（CoRL 2026）

> 来源归档（ingest · 项目页 PDF + 站点摘要）

- **标题：** Pre-training Visual Dexterity in Simulation
- **类型：** paper / dexterous-manipulation / imitation-learning / sim2real / teleoperation / diffusion-transformer
- **会议：** CoRL 2026（PDF 元数据 *Proceedings of the 10th Conference on Robot Learning*）
- **项目页：** <https://spd.bot/> — 归档见 [`sources/sites/spd-bot.md`](../sites/spd-bot.md)
- **PDF：** <https://spd.bot/assets/paper.pdf>
- **arXiv abs：** <https://arxiv.org/abs/2608.15917>
- **arXiv HTML：** <https://arxiv.org/html/2608.15917>
- **代码 / 数据：** 论文宣称释放 **spd-vr**、**spd-75h** 与六套场景；截至 2026-09-05 项目页 **未列** GitHub / Hugging Face
- **作者：** Sarthak Kamat\*、Adam Rashid\*、Satvik Sharma、Aseem Doriwala、Chelsea Finn、Phillip Isola、C. Karen Liu（\* 同等贡献）
- **机构：** 斯坦福大学（Stanford）；麻省理工（MIT）；Scale AI
- **通讯：** sartk@cs.stanford.edu、abrashid@mit.edu
- **入库日期：** 2026-08-17
- **一句话说明：** 全部预训练数据在仿真 VR 遥操作里采（5 人一周 75 h），222M 扩散 Transformer 预训练后，56-DoF 双臂灵巧手每任务 1–2 小时真机微调，五项接触丰富任务均胜过从零 BC。

## 开源状态（步骤 2.5）

- **核查日：** 2026-09-05，打开 <https://spd.bot/> 与 <https://arxiv.org/abs/2608.15917>。
- **已发布：** 项目页、论文 PDF、arXiv 预印本、任务视频与进度图。
- **未发布：** GitHub、数据集下载、权重。
- **论文承诺：** §1 末「we release our pre-training dataset (spd-75h), VR teleoperation software (spd-vr), and six curated scenes…」
- **结论：** **宣称将开源 / 待核实**。wiki 不得写「已开源」；`## 源码运行时序图` 写 **不适用**。GitHub 检索 `spd-vr` / `spd-75h` 无官方仓。

## 摘录 1：仿真 on-embodiment 预训练，而不是人视频或真机遥操作

夹爪路线已靠大规模多任务演示把微调变便宜；多指手仍数据饥渴——真机遥操作吞吐低、UMI 类手持接口难覆盖全部手指 DoF、人视频 off-embodiment 且接触遮挡导致位姿估计噪声、即便手套动捕也有接触点与驱动差异。SPD 用 **spd-vr** 在 MuJoCo 里直接控目标双臂灵巧手：头显 60 Hz 跟踪腕与指尖 → IK 驱动臂/指；仿真 480 Hz；物体全虚拟、接触物理仿真。无真机依赖 → 瞬间 reset、并行、异地采集。五名操作员一周采约 **2,000 / 1,930** 条、**75 小时**（附录 Table 2 合计 1,916 ep / 4,516 min），六场景：拼字积木、餐盘、马克杯、瓶子、杯子、Jenga。任务只规定结局、不锁死策略与子任务顺序。

**对 wiki 的映射：**
- [paper-spd](../../wiki/entities/paper-spd.md)
- [teleoperation](../../wiki/tasks/teleoperation.md)
- [dexterous-data-collection-guide](../../wiki/queries/dexterous-data-collection-guide.md)
- [egoscale](../../wiki/methods/egoscale.md) — 人视频 off-embodiment 对照
- [data-gloves-vs-vision-teleop](../../wiki/comparisons/data-gloves-vs-vision-teleop.md)

## 摘录 2：扩散 Transformer + 历史条件化，短 chunk 才能又稳又反应

策略是 **222M** 扩散 Transformer，flow-matching 速度预测；无语言标注，条件于 visuomotor 历史而非 prompt。输入交织本体（56-D）、上一动作、多视角图像与加噪未来 chunk。训练序列 **256 步 @ 30 Hz**（约 8 s）；因果 mask 下并行去噪全部 chunk。视觉：冻结 **DINOv3 ViT-B/16**，每相机 4 个 query pooling，每两层再 cross-attend 回 patch；图像每 8 步采样。滑窗注意力 **w=32**，部署 rolling KV cache。选用 **c=8** 步 chunk（约 0.27 s）以保持接触反应。优化：Muon（矩阵）+ AdamW，lr \(10^{-3}\)，170k step，EMA。真机：两台升级 **YAM Pro** + 各 **22-DoF Sharpa Wave**（双手+双臂共 **56 DoF**）；三台 RealSense D405（顶 + 双腕尺侧）。真机遥操作对齐仿真 IK，但手指改 **Manus 手套**、腕改 Quest 手柄（头显自带手跟踪不够稳）。微调每任务约 44–121 分钟演示、6k–10k step。

**对 wiki 的映射：**
- [diffusion-policy](../../wiki/methods/diffusion-policy.md)
- [action-chunking](../../wiki/methods/action-chunking.md)
- [behavior-cloning](../../wiki/methods/behavior-cloning.md)
- [paper-why-action-chunking-improves-bc](../../wiki/entities/paper-why-action-chunking-improves-bc.md)
- [imitation-learning](../../wiki/methods/imitation-learning.md)

## 摘录 3：五项真机均胜过从零；历史使短 chunk 成为最强配置

评测：每 checkpoint 每任务 20 trials，按阶段量规打分后归一化成进度。物体与仿真相似但不相同。Table 1（平均进度 %；选用 **w=32, c=8**）：

| 配置 | plates | mugs | jenga | cups | bottles |
|------|--------|------|-------|------|---------|
| SPD w=32 c=8 | **80.6** | **93.3** | **85.0** | **55.6** | 68.8 |
| SPD w=32 c=32 | 44.4 | 78.3 | 16.7 | 36.9 | **70.0** |
| SPD w=1 c=32（π0 风格） | 55.6 | 58.3 | 5.0 | 15.0 | 36.2 |
| SPD w=1 c=8 | 31.9 | 35.0 | 0.0 | 0.0 | 0.0 |
| BC from-scratch w=32 c=8 | 66.9 | 80.0 | 65.0 | 35.0 | 47.5 |

单帧时缩短 chunk 会抖到崩；加 32 步历史后，8 步 chunk 在两种训练体制下都最强——时序一致性来自上下文、反应来自短计划。该配置预训练收益也最大：平均进度相对从零 **+18 点**，其余变体 **≤3 点**。局限：仿真接触参数必须调到操作员能做出「像真的」策略；当前场景/物体覆盖有限，真机评测物体接近仿真；未来可混真机遥操作与 egocentric 人视频，或用 RL 在仿真里继续放大。

**对 wiki 的映射：**
- [paper-spd](../../wiki/entities/paper-spd.md)
- [sim2real](../../wiki/concepts/sim2real.md)
- [paper-teledexter](../../wiki/entities/paper-teledexter.md) — 同系 Sharpa Wave，真机 co-tracking 对照
- [paper-pi05-open-world-vla](../../wiki/entities/paper-pi05-open-world-vla.md) — π0 风格单帧长 chunk 对照

## 当前提炼状态

- [x] 论文摘要填写
- [x] wiki 页面映射确认
- [x] 关联 wiki 页面的参考来源段落已添加 ingest 链接
