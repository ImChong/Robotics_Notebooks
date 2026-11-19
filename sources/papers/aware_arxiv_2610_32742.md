# AWARE：只依赖外部视觉的动力学异常识别（arXiv:2610.32742v1）

> 来源归档（论文一手精读）；研究论文与 Autane 港口自主 RTG 系统不同，不应合并为同一实体。

- **完整标题：** Adaptive World-Models for Anomaly Recognition (AWARE): A Framework for Robot Dynamics Prediction and Introspection
- **作者：** Luke Beddow、George Mavroghenis、Rodrigo Chacon Quesada、Kamil Dreczkowski、Cong Sun、Jiankai Wang、Oscar Kwong-Fai Pang、Antoine Cully
- **单位：** BOLD / Imperial College London
- **版本：** arXiv:2610.32742v1，2026-10-23；CoRL 2026 Poster
- **原文 HTML：** <https://arxiv.org/html/2610.32742v1>
- **PDF：** <https://arxiv.org/pdf/2610.32742v1>
- **项目：** <https://adaptive-intelligent-robotics.github.io/aware/>
- **源码：** <https://github.com/adaptive-intelligent-robotics/aware>
- **类型：** paper
- **原入库日期：** 2026-10-29
- **本次核对：** 2026-11-19
- **核对范围：** 已通读 arXiv v1 HTML 正文摘要、§1–7、Fig. 1–6 与 Table 1、Appendix A–D（含 Table 2–6、Fig. 7–10、算法/超参数/消融/异常判定），并核对官方仓库 README 与 scripts/generate_data.py、scripts/train.py、scripts/evaluate.py 及 configs/*.yaml。原文 HTML 包含论文正文和附录；未宣称运行 GPU 实验或实测复现。
- **通读：** 已完成

## 原文方法摘录

1. **输入与系统定义（§3.1）：** 对参数化系统状态 \(s_t\)、动作 \(a_t\)、建模的内禀参数 \(\gamma_t\) 与未建模扰动 \(\zeta_t\) 建模；输入历史 \(\tau_t=(s_{t-H:t},a_{t-H:t-1})\) 及当前动作。异常分别表现为建模参数变化或未建模外力/碰撞。
2. **动力学估计（§3.2）：** 仿真对电机强度/转动惯量和负载质量做 domain randomization；教师 \(h_\psi(\gamma)\) 提供潜在动力学表示，学生 \(g_\phi(\tau_t)\) 从历史估计 \(\hat z_t\)；不像原始 ROA 为控制策略提供潜变量，AWARE 用世界模型预测监督，并移除潜变量瓶颈/正则，新增参数解码器供故障定位。
3. **世界模型（§3.3）：** 对状态历史、当前动作和 \(z\) 条件化，预测下一状态的高斯分布及方差；论文使用 Transformer 与 RSSM 两种骨干，负对数似然训练。
4. **双通道异常检测（§3.4）：** 计算预测残差按模型标准差归一化并做时间窗口平均，得到 \(E_{pred}\)；潜变量与正常参考分布的 Mahalanobis 距离形成 \(E_{lat}\)。运行时使用固定名义动力学 \(\hat z_0\) 条件化世界模型，有独立阈值，并以 OR 决策联合异常。
5. **互补机制（§5、Appendix C/D）：** \(E_{lat}\) 更擅长长期、建模范围内的电机退化且抗视觉噪声；\(E_{pred}\) 更擅长碰撞等未建模瞬态扰动与严重 OOD 电机衰减；只用一个分数会遗漏另一类故障。

## 原文实验摘录（§4–5、Table 1、Appendix A–D）

- **真实平台与采样：** 人工操作的欠驱动小型起重机，3 个主动轴（slew、luff、hoist）、共 7 关节状态；20 Hz。5 位操作员，在 3 个随机航点间搬运负载。
- **数据组成：** 240 条真实轨迹：80 条正常、40 条负载/钢索碰撞、120 条电机退化。每个电机的 40 条覆盖 75%、50%、25%、0% 速度强度；前两个落在训练随机化 50%–150% 范围内，后两个 OOD。
- **视觉状态估计：** 高噪声模式由外部 ZED X、臂端 ZED 2i + CNN 与两台 Leap Motion 估计状态和操作动作；另设高质量运动捕捉/编码器对照。故障识别评测故意排除了摄像头遮挡等感知失效轨迹。
- **仿真超参数（Appendix A）：** 历史 \(H=64\) 步，预测残差窗口 \(N=20\) 步；电机强度缩放 0.5–1.5、转子惯量 0.1–2.0、负载质量 0.1–2.0 kg；训练使用模拟传感器噪声。
- **结果（Table 1）：** Transformer AWARE 在低噪声模式 AP=0.93、F1=0.87；纯相机高噪声模式 AP=0.77、F1=0.68。对照 RWM 双信号高噪声 AP=0.61，提升约 26% 是相对提升，非绝对 26 个百分点。RSSM AWARE 高噪声 AP=0.76、F1=0.69。
- **故障归因（§5）：** 高噪声条件下电机定位正确率 88.0%，参数估计 MAE 18.1%；低噪声为 93.3%、16.7%。
- **统计与指标：** 采用 AP 与 F1，按连续报警形成轨迹级判断；10 次重复、5 折交叉验证阈值选择、Wilcoxon 检验和 Bonferroni 校正（§4）。不是工业安全认证或跨平台泛化的证明。
- **消融（Appendix C/D）：** 去潜表示/不用方差归一化/不固定名义潜变量均会影响预测异常信号；Mahalanobis 与 OR 结合优于多数简单距离组合，解码器主要提供可解释性。

## 原文明确的局限（§6）

- 正常参考分布假定工况不变，环境或作业条件的正常变化也可能触发误报。
- 真实验证仅一类小型起重机，不能直接外推到大型港口 RTG 或其他机器人。
- 视觉管线排除了严重遮挡与出视野失效；评测尚未覆盖这些感知异常。
- 部署时需要先收集名义参考轨迹来标定阈值，依赖观测质量和运行场景。

## 开源与复现核对

官方仓库有实际源码而非占位：Hydra 配置、MuJoCo MJX 数据生成、RSSM(JAX)/Transformer(PyTorch) 训练、双判别器评测；发布 240 轨迹高/低噪声版本和预训练 checkpoint。核心 README 入口：

```bash
python scripts/generate_data.py num_episodes=10
python scripts/train.py model=rssm_aware dataset_path=data/sim/<date>/<time>
python scripts/train.py model=transformer_aware dataset_path=data/sim/<date>/<time>
python scripts/evaluate.py checkpoint=checkpoints/rssm_aware_seed1
```

数据生成脚本调用 `aware.data_generation.generate_dataset`；训练脚本由 Hydra 初始化 `cfg.model` 和 `cfg.trainer`，调用 `trainer.train()`；评测入口通过 `Agent_Double_Discriminator` 加载模型，经 `aware.evaluation.anomaly.load_dataset` 和 `anomaly.evaluate` 输出 AP/AUROC/F1 等。README 要求 NVIDIA CUDA；该核对不是执行实验。仓库代码 MIT 许可、公开数据 CC BY-NC-SA 4.0，使用时需区分授权。

## 对 wiki 的映射

- [AWARE 唯一论文实体](../../wiki/entities/paper-aware.md)
- [Autane：自主 RTG 起重机（不同论文）](../../wiki/entities/paper-autane-autonomous-crane.md)

## 参考来源

- [arXiv v1 全文 HTML](https://arxiv.org/html/2610.32742v1)
- [官方项目页](https://adaptive-intelligent-robotics.github.io/aware/)
- [官方实现 README](https://github.com/adaptive-intelligent-robotics/aware)
