# OpenNWM：以潜动作预训练构建通用导航世界模型（arXiv:2611.01844v1）

> 来源归档（paper）；本页所有论文公式、论文实验数据以 arXiv v1 为准，不以后续代码更新覆盖论文结论。

- **标题：** OpenNWM: A Generalist Navigation World Model with Latent Action Pretraining / 基于潜动作预训练的通用导航世界模型
- **作者：** Dujun Nie、Ruiqi Song、Zixuan Wang、Hanqing Wang、Yifan Han、Baiyong Ding、Wenzhao Zheng、Hangbin Wu、Long Chen
- **版本：** arXiv:2611.01844v1，2026-11-01
- **原文：** https://arxiv.org/html/2611.01844v1
- **PDF v1：** https://arxiv.org/pdf/2611.01844v1
- **项目页：** https://openspace-lab.github.io/OpenNWM/
- **官方代码：** https://github.com/OpenSpace-Lab/OpenNWM
- **模型：** https://huggingface.co/LittleB0sS/OpenNWM
- **数据：** https://huggingface.co/datasets/LittleB0sS/NavAnywhere
- **Go2 文档：** https://github.com/OpenSpace-Lab/OpenNWM/blob/main/GO2_DEPLOYMENT.md
- **核对范围：** arXiv HTML 2611.01844v1 正文 §1–5、公式 (1)–(9)、Table 1–5、附录 A–G（含数据统计 Table 6、模型及实验细节、扩展实验）；另核查官方项目、仓库 README、lam/README.md、nwm/README.md、实际调度入口及 Go2 文档。
- **通读：** 已完成

## 问题与贡献

传统导航世界模型需要可执行动作标注，跨场景数据收集昂贵；无动作第一视角视频更广泛，却缺少可直接用于动作条件建模的动作标签。OpenNWM 提出：
1. **UNLA**：先从图像对推断连续导航潜动作，用视觉重建和少量真实位姿监督联合约束，减少潜变量仅编码外观变化的问题。
2. **NavAnywhere**：15 个异构无动作视频来源，包含 70,756 段序列、17,496,570 帧、1,188.65 小时源视频（v1 Appendix A，Table 6），覆盖地面机器人、人类与无人机；15 个源均以无动作 RGB 使用，不应把物理标注数据计入其中。
3. **两阶段世界模型训练**：在无动作数据上用潜动作预训练 CDiT；后续以 RECON、HuRoN/SACSoN、SCAND、TartanDrive 的物理位姿动作进行真实动作适配。
4. **导航规划**：后训练世界模型评估真实 waypoint 动作候选并用 CEM 选取动作，在线优化变量不是 UNLA 本身。

## 原文关键公式（公式编号保持 v1）

设 x_tau、x_m 为两帧 RGB 图像，m=tau+k，q_phi 为潜动作后验，z 为连续潜动作，y 为带标签时的物理位姿序列。

**公式 (2)：视觉变分目标**

$$
\mathcal{L}_{\mathrm{vis}}=-\mathbb{E}_{q_{\phi}}\left[\log p_{\psi}(x_{m}\mid x_{\tau},z_{\tau\rightarrow m})\right]+\beta D_{\mathrm{KL}}\left(q_{\phi}(z_{\tau\rightarrow m}\mid x_{\tau},x_m)\|p(z)\right).
$$

其中 p(z)=N(0,I)，视觉重建要求 z 保留转移信息，KL 约束潜空间；视觉预测器是 UNLA 的辅助头，不等于高保真 CDiT 生成器。

**公式 (3)：物理运动监督（仅有动作标注的数据）**

$$
\mathcal{L}_{\mathrm{phy}}=-\mathbb{E}_{q_\phi}\left[\log p_\omega(\mathbf y_{\tau:m}\mid z_{\tau\rightarrow m})\right].
$$

**公式 (4)：联合潜动作学习**

$$
\mathcal{L}_{\mathrm{latent}}=\mathbb{E}_{(\mathbf x,\mathbf y)\sim\mathcal D}[\mathcal{L}_{\mathrm{vis}}]+\lambda_{\mathrm{phy}}\mathbb{E}_{(\mathbf x,\mathbf y)\sim\mathcal D_{\mathrm{phy}}}[\mathcal{L}_{\mathrm{phy}}].
$$

这里 D_phy 是带物理监督子集，不能误读为全部 NavAnywhere 都有动作标签。

**公式 (5)：潜动作、预测跨度和扩散时间的条件嵌入**

$$
\xi_{\tau:m}^{(t)}=\psi_z(z_{\tau\rightarrow m})+\psi_k(k)+\psi_t(t).
$$

该条件通过自适应归一化进入 CDiT，历史视觉状态通过交叉注意力进入模型。

**公式 (7)：世界模型目标**

$$
\mathcal{L}_{\mathrm{NWM}}=\mathbb{E}\left[\|s_m-\widehat{s}_m\|_2^2\right]+\lambda_{\mathrm{vlb}}\mathcal{L}_{\mathrm{vlb}}.
$$

**公式 (8)、(9)：真实动作空间规划**

$$
\mathcal E(\widehat{\mathbf s},\mathbf a;s^g,c)=-\mathcal S_{\mathrm{goal}}(\widehat s_H,s^g)+\mathcal C_{\mathrm{nav}}(\widehat{\mathbf s},\mathbf a,c),
$$

$$
\mathbf a^*=\arg\min_{\mathbf a}\mathbb E_{\widehat{\mathbf s}\sim p_\theta(\cdot\mid\mathbf s_{0:\tau},\mathbf a)}[\mathcal E(\widehat{\mathbf s},\mathbf a;s^g,c)].
$$

s^g 是目标视觉状态；C_nav 是论文定义的导航约束代价。论文 §3.4 的 Algorithm 1 采用 CEM（H=8、每步 0.25 秒、N=80 候选、M=3 随机 rollout），只执行最优序列第一步再重规划。该算法描述不代表官方 Go2 线上执行代码已经发布。

## v1 实验（区分指标及比较范围）

### Table 1：4 秒未来图像直接预测

| 模型 | 参数量 M | ID LPIPS↓ | ID DreamSim↓ | ID PSNR↑ | OOD Stanford PSNR↑ | OOD TUM PSNR↑ | OOD Rover PSNR↑ | OOD Office-Go2 PSNR↑ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NWM | 279 | 0.405 | 0.230 | 13.811 | 11.323 | 8.957 | 11.739 | 10.750 |
| NWM†（CDiT-XL + Ego4D） | 1,099 | 0.419 | 0.227 | 13.523 | 11.444 | 8.884 | 11.906 | **11.083** |
| RAE-NWM | 867 | 0.442 | 0.254 | 13.957 | 11.528 | 9.007 | 9.449 | 10.500 |
| OpenNWM | 280 | **0.368** | **0.196** | **14.299** | **11.591** | **9.135** | **12.574** | 11.047 |

**读法**：OpenNWM 在 ID 三个指标和 OOD 的 Stanford/TUM/Rover PSNR 最好；Office-Go2 PSNR 11.047 低于 NWM† 11.083，不能称四个 OOD PSNR 均第一。

### Table 2：导航轨迹误差（低越好；Office-Go2 是 OOD）

| 方法 | RECON ATE↓ | RECON RPE↓ | SCAND ATE↓ | SCAND RPE↓ | Office-Go2 ATE↓ | Office-Go2 RPE↓ |
|---|---:|---:|---:|---:|---:|---:|
| NWM | 1.13 | 0.35 | 1.01 | 0.28 | 3.03 | 0.68 |
| RAE-NWM | 1.36 | 0.37 | 1.14 | 0.28 | 3.01 | 0.67 |
| OpenNWM | **1.12** | **0.35**（并列） | **0.98** | **0.27** | **2.89** | **0.65** |

ATE=绝对轨迹误差，RPE=相对位姿误差。论文使用离线导航序列评估；Office-Go2 数字不是机载闭环控制成功率。

### Table 3 与 Table 4–5：UNLA 与消融

- v1 Table 3：UNLA 图像重建，ID PSNR **21.071**、LPIPS **0.505**，OOD Stanford PSNR **21.912**、LPIPS **0.472**。
- v1 Table 4（缩小的数据配方）：仅 Time 条件 SCAND ATE **1.045**；Time+Latent Action **0.950**。两组来自消融协议，不能混用 Table 2 完整模型的 **0.98**。
- v1 Table 5（同类消融协议）：仅 Action 监督 SCAND ATE **0.981**；Pixel+Action **0.950**；说明视觉重建与动作监督互补。

## 论文实验条件与工程边界

- 论文 §4.1：世界模型 CDiT-B/2 + Stable Diffusion VAE，4 帧上下文及 4 个预测跨度；LAM 与 NWM 使用 8 × NVIDIA L20。
- 论文评测是未来视觉预测、离线 rollouts 和轨迹评测；不能等同真机实时、碰撞率、到达率或机载模型延迟。
- 官方代码公开 LAM / NWM、模型、数据、离线推理与规划入口；Go2 EDU + Jetson 的文档是**部署计划**，明确尚未验证机载性能，且尚无可运行的 Go2 SDK/ROS2 闭环执行节点。
- 复现前必须区分官方发布的 Stage 1、Stage 2 与 LAM checkpoint：导航推理/规划需兼容的真实动作 Stage 2 权重。

## 对 wiki 的映射

- [OpenNWM 论文详情](../../wiki/entities/paper-opennwm.md)
- [导航纵深路线](../../roadmap/depth-navigation.md)
