# SONIC 论文通读（规模化运动跟踪 · 原文视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/sonic-motion-tracking.md`）

- **标题：** SONIC: Supersizing Motion Tracking for Natural Humanoid Whole-Body Control
- **作者：** Zhengyi Luo, Ye Yuan, Tingwu Wang 等（NVIDIA GEAR Lab 等）
- **类型：** paper / humanoid / motion-tracking
- **原文：** <https://arxiv.org/abs/2511.07820>（2026-11 通过 arXiv PDF v4，2026-08-13 版获取全文）
- **一句话说明：** 把动作捕捉跟踪当作可 scaling 的人形预训练任务（1.2M→42M 参数、1 亿+ 帧、21k GPU 小时），用多编码器 + FSQ 统一 token 接入规划器 / VR / 视频 / 文本 / 音乐 / VLA。
- **核对范围：** 通读正文 §1–§3（含结果、方法、消融、统计）与补充材料 S1（网络 / 超参 / 奖励 / 域随机化 / 部署架构 / 规划器 / VR 与 VLA 集成）、S2（失败案例、推扰、OpenHomie scaling）；Table S3 奖励公式与 Table S4 随机化范围、补充视频说明（S3）、Data S1 仅浏览，未逐项核对；未复核引用文献与官方代码实现。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§1）：** 人形控制不 scale 的根因是任务选择：逐任务手工奖励难以泛化；AMP / ASE / CALM 的判别器信号在数据变大变杂时易模式坍塌。选动作跟踪：MoCap 提供逐帧稠密监督，无需奖励工程。
- **规模（§2.1）：** 训练数据 100M+ 帧（50 Hz；源数据约 700 小时，经 GMR + PyRoki 重定向并滤掉不可执行动作后 611 小时、317,189 clips）；128 GPU × 7 天 ≈ 21k GPU 小时。scaling 三轴：数据 4M/10M/22M/100M 帧，模型 1.2M/16M/42M，算力 2/4/16 节点（≈2k/9k/21k GPU 小时）。最大模型在 test-content（OOD）99.6% 成功、23.8 mm MPJPE-L，最小模型 98.0% / 27.7 mm；增益在 OOD 上最明显。
- **评测协议：** test-content（7,016 clips、15 h、182 个训练中完全没有的子类）、test-repetition（9,395 clips、12 h）、外部 PHUMA（68,326 条）。成功判据为根 / 末端高度偏差 > 0.25 m（局部跟踪，非全局位置）。
- **基线对比：** MuJoCo 下 SONIC 98.5%/99.2%/97.2%（content/repetition/PHUMA），BeyondMimic 82.0%/85.4%/73.8%，Any2Track 61.6%/69.4%/78.5%；MPJPE-L 23.7 mm vs BeyondMimic 40.9 mm。作者自注：基线训练数据与重定向不同，反映跨数据集泛化与 scaling，并非数据对齐的公平基准。
- **专家对比：** 速度跟踪 sim-to-sim，SONIC 存活率 98.5% vs OpenHomie 43.0%（OpenHomie 约 2 m/s 后崩，SONIC 到约 4 m/s 近 100%）；OpenHomie 在 8 GPU 后不再随算力提升（图 S6）。
- **实机：** 124 条动作，实机成功 99.2%（123/124）vs 仿真 100%；MPJPE-L 25.7 mm vs 22.3 mm；差距最大在脚部（53.7 vs 29.0 mm），上身几乎无差（22.2 vs 21.8 mm）。实机部署均用 42M 模型。
- **交互规划（§2.2, §3.3）：** 自回归生成式运动学规划器，0.8–2.4 s 片段做 in-betweening，潜空间 masked token 预测（下采样 4，余弦调度解码）；笔记本 < 5 ms、Jetson Orin 约 12 ms，最快 100 ms 重规划；速度指令 0–6 m/s，蹲 / 跪骨盆高度 0.3–0.8 m，爬行 0–0.5 m/s；根轨迹用临界阻尼弹簧模型（位置 5 ln2、朝向 20 ln2）；25+ 技能 / 风格每个仅需一条代表片段、无需重训。
- **多模态（§2.3, §3.4）：** GEM（扩散式生成通才模型）把视频 / 文本 / 音乐转成人体运动，≥60 fps，经人体编码器 ℰh 输入；滑窗 + inpainting 过渡。
- **VR（§2.4）：** 全身遥操作（PICO 头显 + 脚踝追踪 + 手柄，SMPL 流，87 维/帧）走 ℰh；3 点遥操作（头 + 双腕 SE(3)、手指角、腰高、导航指令）由规划器补下肢，走混合编码器 ℰm。
- **通用策略架构（§3.2）：** MDP + PPO；观测 = 10 步本体历史（关节位置 / 速度、根角速度、重力向量、上一动作）+ 运动指令（机器人 / 人体 / 混合三类，局部坐标、6D 旋转）；动作 = 29 DoF 目标关节位置 + PD。三个 MLP 编码器（ℰr / ℰh / ℰm，各 10 未来帧）→ FSQ（2 个 token，默认 FSQ-32-32，即 64 维）→ 控制解码器 𝒟c(z, 本体状态) 与辅助运动解码器 𝒟r(z)。损失 L = Lppo + Lrecon + Ltoken + Lcycle（重建、编码器两两对齐、循环一致）；非对称 actor-critic（critic 看特权状态）；FSQ 用 straight-through；按失败率加权的 bin 自适应采样（bin 1 s）。训练超参：每 GPU 4096 env、rollout 24 步、5 epoch、clip 0.2、γ 0.99、λ 0.95、actor lr 2e-5。
- **域随机化（S1.1）：** 摩擦 / 恢复系数、初始关节位置、质心位置、周期性根速度推扰、运动指令扰动。
- **消融（§3.6, 表 3）：** FSQ vs VQ-VAE：test-content 成功 99.3 vs 98.7%，MPJPE-L 26.6 vs 35.3 mm（差 8.7 mm，FSQ 避免码本坍塌）；FSQ 容量越大越好且维度比级数影响大（32 GPU 扫描）；三个编码器成功率均 > 99.2%，人体编码器仅比机器人编码器差 0.6 mm，混合编码器差 2.7 mm（稀疏观测）；去掉一致性损失使跨编码器散度增至 8 倍（图 S7）。
- **VLA（§2.5, 表 1）：** GR00T N1.5 在遥操作数据上微调，输出 78 维（64 维 token + 14 维手指）；5 个任务平均 75%（苹果入盘 90%、胡萝卜 75%、刷子 95%、踩垃圾桶踏板 70%、易拉罐入垃圾桶 60%、电钻装箱搬运 70%；每任务 10–20 次试验）。动作空间消融：FSQ token vs SMPL 姿态，平均 68% vs 27%（易拉罐任务 60% vs 0%）。
- **部署（§3.5, S1.2）：** Jetson Orin + TensorRT，策略前向 1–2 ms；四环多速率：策略 50 Hz、指令写入 500 Hz、输入 100 Hz、规划 10 Hz；“最新数据优先”缓冲。安全：3 秒插值到站姿、关节速度看门狗、编码器输入丢失自动降级、停止时纯阻尼（Kp=0, Kd=8）、CRC32 校验。
- **鲁棒性（S2）：** 约 11 kg 物体从头顶上方砸落，机器人无恢复模块仍保持平衡并继续跟踪。
- **局限（§2.6）：** 缺少对安全性与长时间部署能耗的形式化处理；极端或极动态动作下可能失衡；对规划器噪声的鲁棒性依赖指令域随机化与弹簧滤波。
- **作者对“为何跟踪可 scale”的解释（§2.6）：** 逐帧稠密监督在数据变大变杂时仍有信息量，判别器类方法则反馈变弱。（这是作者归因，论文未做专门因果实验。）

## 对 wiki 的映射

- [`wiki/methods/sonic-motion-tracking.md`](../../wiki/methods/sonic-motion-tracking.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2511.07820>
- 项目页：<https://nvlabs.github.io/GEAR-SONIC/>
- 代码：<https://github.com/NVlabs/GR00T-WholeBodyControl>
