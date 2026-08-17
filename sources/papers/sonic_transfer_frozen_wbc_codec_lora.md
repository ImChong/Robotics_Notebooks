# Cross-Embodiment Transfer of a Frozen Humanoid Whole-Body Controller via Analytic Codec and LoRA Adapters（draft v0.1，2026-08-16）

> 来源归档（ingest · 项目页 PDF）

- **标题：** Cross-Embodiment Transfer of a Frozen Humanoid Whole-Body Controller via Analytic Codec and LoRA Adapters
- **类型：** paper / humanoid / whole-body-tracking / cross-embodiment / PEFT
- **项目页：** <https://sonic-agibot-x2.github.io/sonic-transfer/> — 归档见 [`sources/sites/sonic-transfer-github-io.md`](../sites/sonic-transfer-github-io.md)
- **PDF：** <https://sonic-agibot-x2.github.io/sonic-transfer/static/pdfs/paper.pdf>
- **arXiv：** 入库时 **暂无编号**（draft preprint, 2026）
- **作者：** Sitarama Chekuri；Claude Fable 5（Anthropic，AI co-author）
- **机构标签：** Anthropic（AI 共著标注）；目标机 **智元 AgiBot X2 Ultra**；源平台 **NVIDIA GEAR-SONIC / Unitree G1**
- **入库日期：** 2026-08-17
- **一句话说明：** 冻结公开发布的 GEAR-SONIC（G1）全身跟踪控制器，用闭式关节 codec + 动力学解码器 LoRA（约 0.25% 参数、约 2% 平台 cited 算力）迁到 AgiBot X2 Ultra；OOD PHUMA 成功率 **69.0% vs 原生 incumbent 59.0%**，in-dist 无法区分。

## 开源状态（步骤 2.5）

- **核查日：** 2026-08-17。打开项目页 pill 与站点仓 `sonic-agibot-x2/sonic-agibot-x2.github.io`。
- **已发布：** 项目页、draft PDF、演示视频、HF 权重、MuJoCo play bundle [`meetsitaram/sonic-x2`](https://github.com/meetsitaram/sonic-x2)（`./play_v2.sh` 默认真 transfer ONNX + codec sidecar）。
- **边界：** play 仓 **无 LICENSE**；**训练/适配脚本不在该 bundle**；完整 X2 部署栈另见 [`meetsitaram/GR00T-WholeBodyControl-X2-review`](https://github.com/meetsitaram/GR00T-WholeBodyControl-X2-review)。论文写「release models, codec, evaluation protocol」；物理真机验证文中写 **ongoing**。
- **结论：** **部分开源、推理可运行**。wiki 时序图覆盖 `play_v2.sh` 回放路径，不假装可复现 LoRA 训练。

## 摘录 1：问题与五条贡献（Abstract / §1）

Foundation-scale WBT 已公开，但仍焊在源机体上。目标机 incumbent 是约 **1,500–1,600 GPU-h** 的三阶段课程（2k loco → 30k → 130k clips）。本文问更强问题：**预训练平台完全不训，新机能走多远？**

贡献：

1. 冻结全身控制器的跨具身配方：解析 codec + 解码器侧 LoRA，约 **2%** 平台 cited 算力，OOD 上超过原生 tracker。
2. in-distribution 基准对这一差距 **盲**，因此发布 **三角色评测协议**（novel500 / hard300v3 / PHUMA）。
3. 冻结平台下的 last-mile 训练弧：OOD 成功率有斜率、峰值、过训下降 → **gate-and-stop**。
4. 两条边界：跨谱系不变的 **embodiment cost floor**；因果验证的 **信息瓶颈**（奖励看不见的信号无法被 freeze 下游恢复）。
5. 配方双向：去掉 codec 后，同一 adapters 在 **G1 本体**上专精演示库且 **无遗忘**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-sonic-transfer.md`](../../wiki/entities/paper-sonic-transfer.md)；对照 [Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[跨具身迁移选型](../../wiki/queries/cross-embodiment-transfer-strategy.md)。

## 摘录 2：与 Any2Any 的三点差别（§2.1）

二者都迁 **同一份 GEAR-SONIC**，都是「运动学对齐 + 动力学 LoRA」。差别：

| 轴 | Any2Any | 本文 |
|----|---------|------|
| 对齐 | 跨 19–31 DoF、两族形态 → **可学习两级对齐** | 近亲骨架、关节一一对应 → **闭式 affine 表** |
| 冻结边界 | 按机对选模块，含 actor/critic；对齐本身可学 | **最大冻结**：编码器 / FSQ / 运动学解码器 bit-identical；**仅一个 decoder** 插 LoRA |
| 主张 | 低成本 **parity** | 近亲对上 OOD **reversal**（非 strawman：incumbent ~1,600 GPU-h） |

零样本 codec-only 能站/走/舞，是 **形态相似的前置测量**，不是方法主张。Any2Any 所说「即使相似人形也不能直接部署」在关节一一对应的极端端被限定。

## 摘录 3：方法栈（§3–4）

- **平台：** GEAR-SONIC，G1 29-DoF，100M+ MoCap 帧；三编码器 → FSQ token → 运动学 / 动力学双解码器；发布 checkpoint ~**26M** 策略参数。
- **目标体：** AgiBot X2 Ultra；树与关节命名对齐 G1；执行器弱约一倍（踝/腰每公斤力矩），腕更弱。
- **Codec：** 成对重定向语料拟合的 **per-joint affine**；含腕轴命名交换、步长尺度、高度 affine；共享子空间上 encode/decode 互逆；导出为 **calibration sidecar 表**，不是网络。平台只看见 G1，机器人只收到 X2。
- **LoRA：** 只挂动力学解码器线性层；**零初始化**；唯一可训参数 ≈ **0.25%**。Isaac Lab on-policy RL，目标机重定向语料经 codec 喂入。
- **为何不把校正放在输出侧：** companion 冻结规划器可以（下游 tracker 吸收误差）；闭环 WBC 的校正依赖姿态/接触/速度，必须进 decoder 隐特征。companion 自己的结论是 offline fit ≠ closed-loop viability。
- **日程：** breadth（全语料 + 自适应采样，in-dist 平台停止）→ polish（均匀覆盖 + 小份额演示库 rehearsal）。双裁判：IsaacLab 严格跟踪门 + MuJoCo survival。OOD 连续两闸下降即停。
- **原生变体：** 去掉 codec，同一 adapters 专精 G1。

## 摘录 4：数字与边界（§5–9）

**零样本（codec only）：** 轻松 in-regime 关节误差可比 incumbent（放松走 6.1° vs 8.6°），但观感更抖；200-clip 生存约 **75% vs 98%**，失败集中在骨盆下沉与单支撑——几何过了，动力学没过。

**主表：** 见项目页 / wiki。in-dist ~96%/33 mm 不可分；OOD 十个成功点；breadth 阶段已超过 incumbent。选中 checkpoint 全谱系约 **135 GPU-h**（8 卡一夜）；含失败分支 <190。相对平台 cited ~2%；相对 incumbent ~1/8。

**过训弧（§6）：** polish 期 OOD 约 **+2.7 成功点 / 千 iter**，峰值后四闸单调下降（69.0 → 64.4），in-dist 持平；幸存者误差略好 = 专精吃泛化。训练 reward 无预警。停止规则：两闸下降选峰。

**Embodiment floor（§7）：** 所有 X2 模型 OOD 均值误差落在 **41–43 mm**，同平台在 G1 上 **32 mm**。训练改覆盖率，不改地板。归因：执行器 gap + 重定向（肘需多屈以匹配手位）。

**信息瓶颈（§8）：** 发布训练的关键点止于腕 link、奖励无关节空间跟踪；腕 roll 绕自身关键点转，奖励看不见。参考扰动半弧度：肘 **73%** 传到命令，腕 **11%**（近噪声）。部署把腕钉在固定姿态。

**G1 本体专精（§9）：** 演示库 93.6% → 饱和恢复；guard novel500 **98.6%/23.0 mm vs 发布 98.2%/25.7 mm**；加倍预算无过训弯折。

**局限（§10）：** 一对近亲机器人；OOD 集兼作停止信号（报告数字来自另抽 disjoint sample）；过训机制未消融；选中 checkpoint 的硬件证据是 **双仿真裁判**，真机验证 ongoing。

## 对 wiki 的映射

- 沉淀实体页：[SONIC-Transfer](../../wiki/entities/paper-sonic-transfer.md)
- 交叉：[Any2Any](../../wiki/entities/paper-any2any-cross-embodiment-wbt.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)、[跨具身迁移选型](../../wiki/queries/cross-embodiment-transfer-strategy.md)、[WBT pipeline](../../wiki/concepts/whole-body-tracking-pipeline.md)、[LoRA](../../wiki/concepts/lora.md)、[PHUMA](../../wiki/entities/dataset-bfm-phuma.md)
