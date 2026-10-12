# CASTER 论文来源归档

## 元数据

- **题目：** CASTER: Formalizing Zero-Shot Manipulation from Synthetic Videos as Constrained Optimization
- **作者：** Yinlong Dai, Homanga Bharadhwaj, Dylan P. Losey
- **单位：** Collab 实验室，Virginia Tech 机械工程系；Johns Hopkins University 计算机系（B3）
- **预印本：** https://arxiv.org/abs/2610.12567
- **HTML：** https://arxiv.org/html/2610.12567v1
- **版本：** v1，2026-10-08，cs.RO（交叉 cs.CV）
- **项目页：** https://collab.me.vt.edu/CASTER （归档见 sites/caster-vt-collab-project.md）
- **代码：** https://github.com/VT-Collab/CASTER （MIT；归档见 repos/vt-collab-caster.md）
- **数据集：** https://huggingface.co/datasets/JiecaoBoy/CASTER-scenes （项目页链接，本次未逐文件核对）
- **读取范围：** arXiv 摘要页与 HTML 全文抽取；附录有读到，未逐页核对公式推导。

## 论文要解决的问题

视频生成模型可以在机器人自己的场景里合成任务视频，无需真人示教。但生成视频看似合理，却不满足物理一致性，也不绑定机器人本体，机器人不能直接照抄其中的运动。论文把「合成视频 → 零样本可执行轨迹」形式化为带约束的优化问题：从视频提取的物体轨迹只当作搜索初值，再在数字孪生里满足物理可行性与语义一致性后精修。

## 方法与训练信息

1. **形式化：** 变量为机器人动作轨迹 ξa（诱导物体轨迹 ξo）；目标是最小化 ‖ϕ(ξo) − ϕ(ftrack(v))‖，即匹配任务特征而非逐点路标；约束为语义概率 P[𝒢τ(ξ)=1] ≥ 1−ε_sem 与物理可行概率 P[ξ∈ℱℛ] ≥ 1−ε_phys，数字孪生中另加安全裕度 δ。
2. **语义吸引域：** 𝒮ρϕ = {z : dϕ(z, ℳsemϕ) ≤ ρ}；用 Markov 不等式给出 P[z∈𝒮ρϕ] ≥ 1 − σvid²/ρ²，说明视频生成器需要概率性地把搜索放进语义吸引域，轨迹才能被局部调整而不丢任务。
3. **实际目标（Eq. 19）：** 最小化 Σt wt‖ϕ(ξot) − ξ̄τt‖，约束接触力 ‖F‖ ≤ F̄ 与关节限位。论文声称既有方法（如 retarget、mimic 类）是该框架在关键简化下的特例。
4. **流水线：** HunyuanVideo-1.5 480P-I2V Step-Distilled 每任务生成 20 段视频（seed 0–19，121 帧、24 fps）；Qwen3.5 作 VLM 判官，保留分数 ≥0.75 的视频；Depth Anything 3 估深度（对首帧做仿射对齐）+ TrackCraft3R 做 3D 点跟踪；LLM 提议目标–上下文物体对与轨迹维度，跨视频取中位数/均值，权重随跨视频方差指数衰减。
5. **数字孪生与优化：** Isaac Sim 6.0.1 + Isaac Lab，SAM 3D 重建物体；物理 120 Hz、控制 60 Hz、1050 个并行环境。优化器为样条控制点上的交叉熵法（CEM）：附录给 (Kpos, Krot) = (15, 5)，种群 1024，10 代，精英 10%（102 个）。
6. **执行：** 阻尼最小二乘微分 IK，开环执行。

## 论文报告的结果

- **平台与任务：** Franka Emika 机械臂 + 固定 Orbbec Femto Mega RGB-D 相机；12 项任务（香蕉入锅、倒杯、方块入篮、倒水壶、橙子装盘、堆方块、叠碗、盖桶、合盖、无碰撞抓取、烤面包、白板画线）。Retarget 每任务 7 次；Mimic 与 CASTER 每任务 21 次（7 次 CEM × 3 条轨迹）。
- **总成功率：** Retarget 32.1%；Mimic 42.5 ± 4.5%；CASTER 76.6 ± 3.9%。摘要中的 “34.1% 平均提升” 对应与最佳基线 Mimic 的绝对差（76.6 − 42.5），不是相对提升。
- **逐任务（Retarget / Mimic / CASTER，%）：** 香蕉入锅 0.0/66.7/100.0；倒杯 42.9/52.4/95.2；方块入篮 28.6/76.2/100.0；倒水壶 57.1/66.7/61.9；橙子装盘 57.1/61.9/100.0；堆方块 0.0/9.5/57.1；叠碗 14.3/28.6/100.0；盖桶 42.9/28.6/33.3；合盖 14.3/4.8/57.1；无碰撞抓取 85.7/90.5/95.2；烤面包 28.6/14.3/90.5；白板画线 14.3/9.5/28.6。
- **视频生成器对比（任务 1、4、6、7，共 80 段）：** HunyuanVideo-1.5 有效 55/80、成功 79.8 ± 6.5%；Wan 2.2 I2V-A14B 71/80、76.2 ± 6.1%；LTX-2.3 22B Distilled 52/80、47.6 ± 8.6%；CogVideoX1.5 5B-I2V 1/80、0.0%。作者结论：有效视频数量不单独决定下游成功率。
- **数字不一致：** 附录称平均 70.4% 的视频通过 VLM 校验，但表中 55/80 = 68.75%，两处不一致，本归档只引表内数字。

## 复现与解释边界

- 倒水壶（61.9 vs Mimic 66.7）与盖桶（33.3 vs Retarget 42.9）CASTER 并非最佳。
- 论文承认仿真到真机差距：壶柄在夹爪内旋转；桶的旋转手柄关节在刚性网格重建中缺失。
- 假设特征跟踪器准确、相机到机器人标定准确；视频生成器冻结；全程开环；生成器对比只覆盖 4 个任务、每任务 7 段视频；Retarget 基线试验次数少于另外两者。
- 依赖多个重型外部模型（HunyuanVideo、Qwen3.5、SAM 3D、Gemini Robotics ER、Isaac Sim），复现门槛高。

## Robotics_Notebooks 映射

- 详情节点：wiki/entities/paper-caster.md
- 汇总页：wiki/overview/embodied-station-34-papers-spatial-grounding-technology-map.md
- 概念入口：wiki/concepts/video-as-simulation.md
- 任务入口：wiki/tasks/manipulation.md

## 一手来源

- arXiv：<https://arxiv.org/abs/2610.12567>
- arXiv HTML：<https://arxiv.org/html/2610.12567v1>
- 项目页：<https://collab.me.vt.edu/CASTER>
- 代码：<https://github.com/VT-Collab/CASTER>
