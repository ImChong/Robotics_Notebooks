# LightNav-0 论文通读

> 来源归档（原文通读 · 支撑 `wiki/entities/paper-lightnav-0.md`）

- **标题：** LightNav-0: Eliciting VLM Spatial Intelligence for Generalist Embodied Navigation
- **作者：** Light Origins Team（亮源新创）
- **类型：** paper / embodied-navigation / vlm
- **原文：** <https://arxiv.org/abs/2608.30935>（2026-11 通过 arXiv PDF v2，2026-09-09 版获取全文）
- **一句话说明：** 以 Qwen3-VL-4B 为骨干，用 dual-channel pointing（affordance + object 两个图像网格 token）加 3 级 RVQ 动作 token 统一指令跟随 / ObjectNav / 视觉跟踪，经 ER mid-training、SFT（含 DAgger）、GRPO 在线 RL 训练，单目 RGB 在 10 个仿真设置上取得论文口径最优。
- **核对范围：** 通读摘要、§I–§VII 正文（架构 §III、数据与 INSIGHT-Bench §IV、训练 §V、实验含消融与缩放 §VI、结论）及全部表格 / 图注；致谢与参考文献仅浏览；该论文无附录。数值表（Tab. III–X）逐项看过但未复现。未阅读代码仓库实现。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **动机（§I、§II）：** 现有导航系统依赖任务 / 本体专用部件（waypoint predictor、拓扑图、动作头）；主张紧凑 VLM 可作共享推理骨干，不加任务头、不加任务标识 token。
- **架构（§III-A）：** 实例化自 Qwen3-VL-4B-Instruct（原生分辨率 ViT + 36 层 LM），仅扩充词表（⟨apos_i⟩、⟨opos_i⟩、RVQ 动作 token）；每步输出 = 1 个 affordance 点 token + 1 个 object 点 token + 3 个 RVQ token（式 4），解码为 10 个未来 SE(2) 路点。
- **视觉历史压缩（§III-B，式 1–2）：** 仿 Ebbinghaus 遗忘曲线，历史帧采样率按年龄指数衰减，池化步长随年龄指数增大；像素预算 256K / 576K / 1M；远期帧 token 更少更粗，当前帧保持最细。
- **Dual-channel pointing（§III-C，式 3）：** 点被编码为单个图像网格 token（i = r·W_g + c），两个通道使用不同 token 族；保留索引表示原地转向、停止、目标不可见；作为“潜在空间 CoT”，长度固定，不产生变长文本推理。
- **RVQ tokenizer（§III-D，式 5–8）：** 3 级码本各 256 项，距离为 ADE + λ·航向误差（λ=0.3）；任意非空前缀均可解码为可执行轨迹，可只用前 1 或 2 级换取更低解码开销；3 级 ADE 0.72 cm，对比单 token K=4096 的 VADv2 式词表 2.48 cm；单进程 vLLM 服务，RTX 4090 上约 4 ms / 生成 token。
- **数据（§IV）：** 导航语料 2K+ 场景 / 4K+ 小时；ER mid-training 36 个来源（pointing 35.14%、单图 VQA 25.05%、视频 19.81%、通用 / 抽象 20%）；SFT = 16 个导航来源 + 33 个 ER/VQA 来源，77.6% 导航监督、22.4% 复习 ER；相机随机化（FoV 90–130°、高 0.5–1.5 m、俯仰 ±15°）；stop 样本上限 2%，轨迹簇占比上限 5%。
- **INSIGHT-Bench（§IV-D）：** 训练 1,683 场景 / 53,090 episode，评测 210 场景 / 1,097 episode，训练评测场景不相交；Molmo2 开放集 pointing + 深度提升 3D + 跨视角聚类（≥2 视角、空间散布 <0.6 m）；指令经规则模板或视频 VLM（Seed2.0）起草、LLM 改写加语义检查、末帧到达验证；二维诊断分类：5 种场景 × 5 种指令类型。
- **训练（§V）：** ER mid-training lr 1e-5、batch 128，约 170 H100 小时，产出 LightNav-ER；SFT lr 1.5e-5、batch 320、序列 8192，约 950 H100 小时，含 DAgger；在线 RL 用 GRPO（G=8、B=32 种子 / 256 episode 每次更新，ε=0.2，β=0.01，lr 1e-6，单节点 8×H100），终端标量奖励分配给所有步，只保留 mixed 难度 episode 为主并保留少量两端；三类任务各有终端奖励（EVT 含可见性 / 位置 / 持续性、VLN 含 nDTW 对齐与到达、ObjectNav 含 PL 效率与接近度）。
- **ER 评测（Tab. II）：** LightNav-ER（4B）8 个基准平均 67.4，对 Qwen3-VL-4B 63.1、Molmo2-ER-8B 62.8；4 项第一、4 项第二。
- **导航评测（Tab. III–VIII）：** 单目设置下 R2R SR 68.5 / SPL 62.8；RxR SR 73.6，但 nDTW 67.4 低于 DualVLN 70.0；ObjectNav MP3D SR 53.3、HM3D v1 74.5、v2 77.2；HM3D-OVON seen / synonyms / unseen SR 55.3 / 54.6 / 47.0；EVT-Bench STT SR 91.7、DT SR 82.6（CoMaTrack 的 STT SR 92.1 略高，ReferTrack TR 更高）；INSIGHT-Bench SR 43.7 / SPL 41.5 / NE 3.88（次优 JanusVLN SR 27.4）。多视角方法在 R2R 等上仍略占优。
- **INSIGHT 细分（Tab. VII、Fig. 12）：** Direction 类 57.7 最高，Extremum 37.2 最难；Apartment 61.1 最好，Institution 29.2 最弱；House–Direction 74.0%，Institution–Extremum 18.0%。
- **跨域与真机（§VI-D）：** 同一权重在 CS 1.6、VizDoom、Minecraft、Trigger Rally 四个游戏域，以及人形 / 四足（Go2）/ 空中 / 轮式（LimX TRON 1）四种本体上演示，经共享 trajectory follower 接入各自内置运动策略；论文明确是定性展示，不作定量对比；追踪监督只含人类目标，演示中也跟随人形机器人、轮式机器人和推车。
- **消融（Tab. IX–X）：** ER 初始化使 8 设置平均 SR 60.5 到 63.0、SPL 38.8 到 39.9；去掉 pointing 监督使平均 SR 54.7 到 63.0、SPL 34.3 到 39.9（HM3D v1 上 +13.5 SR）。
- **缩放（§VI-F，Fig. 16）：** 2B 到 4B 明显提升（R2R SR 59.9 到 68.5），8B 不一致（R2R SR/SPL 反降 1.6/0.5）；数据 1/16 到全量单调但边际递减；环境数量缩放增益大于同量级数据缩放。
- **结论与展望（§VII）：** 当前单一决策通路，未分离高频局部控制与低频语义推理，后续可做双系统；可用互联网视频预训练扩展开放世界概念。

## 对 wiki 的映射

- [`wiki/entities/paper-lightnav-0.md`](../../wiki/entities/paper-lightnav-0.md)
