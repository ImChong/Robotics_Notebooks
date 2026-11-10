# A Unified Inertial Encoder for Human and Humanoid Motion（arXiv:2611.10137）

> 来源归档（paper）

- **标题：** A Unified Inertial Encoder for Human and Humanoid Motion
- **类型：** paper
- **作者与机构：** Yutian Chen、Shibo Zhao、Rocky Duan、Karen Liu、Carmelo Sferrazza、Guanya Shi、Yuheng Qiu、Koushil Sreenath；Amazon FAR（1），UC Berkeley（2），Stanford University（3），Carnegie Mellon University（4）（论文首页）
- **arXiv：** <https://arxiv.org/abs/2611.10137>（v1，cs.RO，2026-11-06）
- **PDF：** <https://arxiv.org/pdf/2611.10137v1>
- **项目页：** <https://unified-inertial-encoder.github.io/>
- **代码：** https://github.com/amazon-far/Unified-Inertial-Encoder
- **入库日期：** 2026-11-10
- **一句话说明：** 用「运动学 IMU 合成」把人体与重定向 G1 人形放进同一 12 部位惯性输入空间，联合训练速度估计与动作-文本对齐，零样本迁移到真机 Unitree G1。
- **核对范围：** 通读 arXiv v1 PDF 全文（摘要、引言、相关工作、方法 III-A 至 III-D、实验 IV-A 至 IV-F、结论、参考文献列表），含图 1-7 与表 I-V 的文字内容；论文无附录 / 补充材料章节（PDF 在参考文献后结束）。另读了项目页文字与 GitHub README（只静态读，未克隆、未运行代码）；README 的实现细节未与代码逐行核对。
- **通读：** 已完成

## 开源状态

- **已开源**（步骤 2.5 核查，2026-11-10）：项目页「Code」链接指向 GitHub 仓库；README 称包含训练、基线对比、数据预处理、评测与论文分析，Apache-2.0 许可，初始发布 2026-11。仓库未克隆，未核实权重是否发布（README 摘录中未见权重说明，推测需自行训练）。

## 核心摘录（均出自论文，括号为章节 / 表号）

1. **问题设定（摘要、I）：** 从机器人录像学人形状态估计与语言动作理解，训练覆盖受限于控制器能执行的动作库，并有「估计器依赖准确估计、控制依赖估计器」的循环依赖；人体数据集覆盖更广但存在身体几何、传感与物理扰动差异。
2. **共享惯性接口（III-A、III-B）：** 12 个解剖对应部位（骨盆、头、左右上臂 / 前臂 / 大腿 / 小腿 / 脚）。人体信号来自穿戴 IMU 或由运动学得到的虚拟传感器；G1 信号由骨盆 IMU + 关节编码器 + URDF 运动学「运动学传输」合成（式 2，含切向、向心、科氏与相对连杆加速度项），不需要世界位姿或根速度；全部信号旋转到 Xsens 坐标约定（式 3）。
3. **连续时间编码（III-B.3）：** 学习的分层连续卷积（线性 B-spline 权重，式 4）按采样时刻聚合，输出定长时间 token，避免重采样；Transformer 交替「站点内注意力」与「全站点注意力」，时间旋转位置编码，[CLS] 汇总用于语言对齐。
4. **联合训练目标（III-C，式 5-7）：** 速度损失 + 短区间位移轨迹一致性损失（区间 0.25 / 1 / 2 s，RK4 积分去噪角速度，对旋转停梯度）+ 与冻结 CLIP ViT-L/14@336px 文本特征的余弦对齐 + 去噪辅助损失；λ_den = λ_lang = 1，速度与轨迹权重按梯度范数比做 EMA 自适应（截断于 [0.02, 50]）。
5. **随机化（III-D，表 I）：** 重力 U(9.76, 9.83)、体型缩放、速度缩放、人体采样率 120-240 Hz、时间戳抖动、传感噪声、振动与冲击；范围「启发式选取」。
6. **数据（IV-A、表 II）：** Nymeria（300+ h，240 Hz，有语言）、Natural Motion（40.6 h，无语言）、BONES-Seed 人体（144 h）与其重定向 G1（144 h，120 Hz）；真机 G1 轨迹仅 0.7 h、200 Hz、无语言，只用于评测。人体数据按轨迹 85% / 15% 划分；G1 的「复杂动作」（如舞蹈）全部留作未见评测子集。Fig. 1 标注总计 500+ 小时、3500+ 种动作。
7. **惯性里程计结果（IV-B，表 III）：** 在五个数据集上 AVE 与 ATE_GT 均最低；AVE：重定向 G1 0.024 m/s，真机 G1 0.099 m/s（次优 AirIO w/ joints 为 0.190）。ATE_Pred 在四个数据集领先，Nymeria 上 TLIO 略好（17.76 对 18.20 m）。论文承认「积分姿态下误差更高，说明姿态漂移仍是局限」。真机 G1 ATE_GT 3.153 m，ATE_Pred 11.50 m。
8. **语言检索（IV-C，表 IV）：** G1 未见类别（仅人体训练中出现）R@1 53.8% / mAP 11.3%，IMU2CLIP 为 17.3% / 0.5%；Nymeria 上 IMU2CLIP R@1 更高（16.2% 对 14.9%），本文 mAP 更高。R@1 的候选为 19 个随机干扰项（8 次平均）。
9. **真机与时延（IV-D、图 5）：** 真机 G1 无需真机训练或微调；TensorRT + NVIDIA L40S 上处理 4 s 窗口、12 个 IMU、200 Hz 输入耗时 1.78 ms。真机语言检索仅给出定性演示（楼梯下行，「Walking」到「Standing」）。
10. **消融（IV-E，表 V）：** 仅人形训练 → 联合训练，真机 G1 ATE_GT 5.901 → 3.153 m（重定向 G1 0.162 → 0.101 m）；单骨盆 IMU 真机 8.114 m；骨盆 IMU + 关节 3.513 m（ATE_Pred 18.36 m，劣于全方法 11.50）；去掉连续编码误差大幅上升（真机 17.37 m）；数据量 1/64 到全量，真机 ATE_GT 4.369 → 3.153 m，重定向 G1 在 1/4 数据时最好（0.099 m）。
11. **算力分配（IV-F，图 7）：** Chinchilla 式拟合 L(N, D)，在 C = 6ND 下 N_opt ∝ C^0.52、D_opt ∝ C^0.48；作者注明含外推、优化的是验证误差，未必给出最佳硬件迁移。
12. **作者自述局限（V）：** 硬件评测只有一个人形平台；语言监督部分为合成且词汇有限；姿态漂移仍是轨迹误差的主要来源；后续计划多平台评测并与滤波结合做闭环状态估计与控制。

## 实现细节（IV-A.3）

- 20 epoch、4096 ms 窗口、bfloat16、batch 1024；Muon（矩阵参数学习率 2e-3）+ AdamW（其余 5e-3）；结果使用权重 EMA 0.95。
- README 补充（非论文正文，未核对代码）：6 个 Transformer block、宽度 768、每 token 125 ms，训练用 8 GPU、每卡 128 窗口。

## 二手说法核对（具身智能小站 2026-11-10 条目）

- 「通过共享惯性接口把人体动作与重定向人形动作放到同一输入空间，同时学习速度估计和动作语言对齐」与论文一致（III）。
- 「可零样本用于 G1」与论文一致（摘要、IV-D），但论文对真机的语言检索只有定性演示，定量语言检索（表 IV）是在重定向 G1 上做的。
- 「姿态漂移仍会累积成轨迹误差」与论文自述一致（IV-B、V）。
- 小站条目未给机构，论文首页为 Amazon FAR 等（见上）。

## 对 wiki 的映射

- [paper-unified-inertial-encoder](../../wiki/entities/paper-unified-inertial-encoder.md)
