# HorizonStream — 论文全文通读归档（arXiv:2605.23889v2）

> 一手阅读笔记；以 arXiv 2026-10-24 v2 全文及附录为基准，论文实验数字属于作者报告，非独立复现。

- **英文标题：** HorizonStream: Long-Horizon Attention for Streaming 3D Reconstruction
- **中文标题：** HorizonStream：用于流式三维重建的长时域注意力
- **作者：** Chong Cheng; Peilin Tao; Nanjie Yao; Guanzhi Ding; Xianda Chen; Yuansen Du; Xiaoyang Guo; Wei Yin; Weiqiang Ren; Qian Zhang; Zhengqing Chen; Hao Wang
- **机构：** HKUST(GZ)、Horizon Robotics、CASIA、CSU（参照原文作者署名）
- **会议：** NeurIPS 2026（arXiv 条目明确标注 Accepted to NeurIPS 2026）
- **arXiv：** <https://arxiv.org/abs/2605.23889>；全文 HTML <https://arxiv.org/html/2605.23889v2>
- **版本：** v1 2026-05-22；**阅读版本 v2 2026-10-24**
- **项目页：** <https://3dagentworld.github.io/horizonstream/>
- **代码：** <https://github.com/3DAgentWorld/HorizonStream>
- **核对范围：** arXiv v2 正文 §1 Introduction、§2 Related Work、§3.1–3.4 Method、§4.1–4.4 Experiments、§5 Conclusion；Appendix A（Geometric Attention Dilution）、B.1–B.4（零遗忘、有效记忆时长、范数界及 TTT 联系）、C.1–C.3（网络、超参数、训练数据）、D（数据集、训练重叠、缺失值统计规则）、E/E.1（额外实验、失败案例、线性探针）；对照 Figures 1–10、Tables 1–8 及官方 README。
- **通读：** 已完成

## §1–2：问题与现有方法的失效机制

- 因果在线 3D 重建需从历史 RGB 流估计相机 SE(3) 位姿与稠密深度；不能使用未来帧，模型持久状态须有界。
- 历史证据不是“越新越有用”：局部 2D/3D 匹配过期快，场景结构与全局尺度应留存更久。
- 滑窗是硬截断核；cache refresh 带来块状不连续；长 causal softmax 易形成 attention sink 且 KV 随 t 增长；未门控 recurrence 使旧错误长期污染状态。
- 与优化式 SLAM、离线 VGGT 及 CUT3R/TTT3R/LongStream 等流式几何网络区分。本文重点是**不同几何证据的影响寿命分解**，不是使用无限历史帧做全注意力。

## §3：方法与公式

1. **证据影响核（§3.1，Eq.1）：** `K(t,i)=K_{\mathrm{spatial}}(t,i)\,K_{\mathrm{time}}(t,i)`。空间项负责短窗可靠匹配；时间项负责有界的跨窗累积；额外 MRT 负责读取度量信息。它是方法的建模分解，非独立可训练的图像分割网络。
2. **Geometric Linear Attention / GLA（§3.2，Eq.2–7）：** 用折扣几何目标解释状态更新，单通道式为 `S_t=\gamma_t S_{t-1}+\phi(k_t)\tilde v_t^\top`，扩展成通道门控 `S_t=\mathrm{diag}(\boldsymbol\gamma_t)S_{t-1}+\phi(k_t)\tilde v_t^\top`，每个通道生命周期 `\tau^{(c)}=-1/\log\bar\gamma^{(c)}`。小 γ 快忘旧匹配，大 γ 保存结构/尺度证据，持久状态是固定尺寸，**不等于存储完整的高分辨率全局地图**。
3. **Gated Window Attention / GWA（§3.3，Eq.8）：** 各注意力头门控 `g_h=\sigma(W_g\bar x+b_g)`，对可能虚假的局部对应降权；Spatiotemporal RoPE 沿时间/高度/宽度三轴注入相对位置信息，时序索引定期重置（不是重置长期几何状态）。
4. **Metric Readout Tokens / MRT（§3.3，Eq.9）：** 每帧度量 token 经长期 GLA 读取尺度；尺度头预测 `\hat s=\exp(g(z^{metric}))`，同时缩放深度与平移。局部 pose tokens 经 Transformer 聚合出共识相对位姿；DPT 预测稠密深度与置信度。
5. **架构/损失（§3.4）：** ViT-L，初始化涉及 VGGT/DINOv2；交替帧内块与跨帧全局块，局部 GWA + 部分层 GLA；pose/depth/scale 联合监督；scale loss 仅在度量标注样本上生效。可选 loop closure 利用 DINOv2 全局描述子候选检索、局部重推理及位姿图优化，属于后处理增强，不应混入无 LC 的纯在线指标。

## §4：训练与评估协议

- **训练：** 每样本 48 帧，逐 21 帧计算块传播 GLA 状态，pose 参考窗口 W=10。两阶段：Stage 1 64×A800 60k iterations；Stage 2 64×H20 40k iterations，加入更多长序列场景但仍只抽 48 帧。AdamW 2e-5、warmup 2000、cosine schedule。共 24 个数据集/来源（Appendix C.3 给具体 stage 比例），含内部长序列数据；**训练代码没有公开**。
- **评估：** KITTI / vKITTI2 / Oxford Spires / ScanNet++ / TUM RGB-D / Waymo / VBR / ETH3D / 7Scenes；表格是各数据集指定协议下的 ATE、CD、F1、depth，不应相互混用。
- **Table 2，无 LC：** KITTI ATE 19.75 m（LongStream 51.90，Lingbot-map 25.29）；Oxford Spires ATE 9.38 m（Lingbot-map 15.46）；表内推理 13.20 FPS。**加 LC：** KITTI 16.44 m、Oxford 8.71 m、10.45 FPS。推理 FPS 是特定论文设置，并非任何设备上的实时保证。
- **Table 3，VBR 7 段长序列：** 无 LC 平均 ATE 25.30 m；有 LC 18.84 m；这些序列包括 10,021–18,846 帧，分别报告每序列结果，不把 10k+ 当作所有数据集均无漂移。
- **Table 4/5：** Oxford Spires 点云 CD 4.97、F1@4 0.89；7Scenes CD 2.98、F1@0.25 0.93；KITTI depth Abs Rel 0.057、δ<1.25 94.8%，后者**并非**比 VGGT 的 97.0% 更好。
- **Table 6，vKITTI2 400 帧消融 ATE：** 完整模型 1.20；去 GLA 5.38；去 channel-wise gate 3.21；去 GWA 7.46；去 GWA head gate 4.06；去 MRT 3.34。这证明各模块重要，但该消融不能直接外推为万帧误差减少相同比例。
- **失败结果处理（Appendix D）：** ATE 表的“–”表示缺失/失败，算均值时用该数据列已报告最大误差做惩罚，不能把该均值误当全成功样本均值。
- **数据重叠（Appendix D）：** vKITTI2 是训练重叠评估；7Scenes 评测的 Chess、Fire、Office、RedKitchen 在训练范围，而 Heads/Pumpkin/Stairs 不在；Waymo 虽为训练数据来源，但评测具体片段 held out；故必须区分 in-domain、scene overlap 与独立未见测试。

## Appendix A–E：理论、配置与边界

- **A：** 当几何共视相关帧个数有界、attention scores 有界时，causal softmax 分给几何相关帧的注意力上界随着总序列增长下降；sink 不能解决 KV 增长或无关帧注意力浪费。
- **B.1–B.3：** γ=1 时初态影响不消退；若所有通道上界 `\bar\gamma<1` 且输入有界，则初态影响指数衰减、状态范数有界。界是**条件性数学结论**，不等于模型在任意传感器噪声与场景下不漂移。3τ 后通道影响约剩 5%。
- **B.4：** 将折扣 TTT 的一步梯度形式联系到 gated linear attention；并非推理每帧都执行完整显式 TTT 训练。
- **C：** 具体架构层、hyperparameters 与各数据集采样权重；48 帧训练与长流推理通过重复持久状态传播连接。
- **D：** 实验序列分割、frame subsampling 禁止及指标失败惩罚的细则。
- **E/E.1：** Oxford Spires 单序列结果，loop closure 改善回环区域，内存/延迟从 200 到 10k 帧的趋势；稠密复访或外观模糊区域仍可能定位错误；线性探针在 KITTI 00/02 拟合、05 测试不同 GLA 层中的几何信息，不是端到端任务成功率。
- **实现与数据边界：** 论文图示的序列长度无关**网络持久状态**不代表磁盘输出、CPU 汇总点云与可选 loop closure 占用常数内存；论文 GPU memory 不包括可选 loop closure，README 约 8.5GB 对应 sliding-size=1 的特定设置。

## 对 wiki 的映射

- [HorizonStream 唯一项目实体](../../wiki/entities/paper-horizonstream.md)
- [作者项目页核验](../sites/horizonstream-project.md)
- [官方实现入口及授权核验](../repos/horizonstream-official.md)
