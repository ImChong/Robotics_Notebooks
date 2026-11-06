# OASIS 论文通读（Loco-Manip 8 篇地图的「仿真入口」一手核对）

> 来源归档（原文通读 · 支撑 `wiki/overview/loco-manip-8-papers-technology-map.md`）

- **标题：** OASIS: From Simulation Data Collection to Real-World Humanoid Loco-Manipulation
- **作者：** Zehao Yu, Jiakun Zheng, Weiji Xie, Jiyuan Shi, Chenyun Zhang, Chenjia Bai, Xuelong Li（TeleAI / 中国电信人工智能研究院等）
- **类型：** paper / humanoid / loco-manipulation / sim-data
- **原文：** <https://arxiv.org/abs/2606.08548>（2026-11 通过 arXiv PDF v1 获取全文）
- **一句话说明：** 全程在仿真中采集人形 loco-manip 演示（真实图 → 生成式 3D 资产 → VR 仿真遥操作记录状态 → 离线路径追踪渲染 + 域随机化扩增），层级 Flow Matching 策略在 Unitree G1 上零样本部署。
- **核对范围：** 通读正文 §1–§6（方法、三组实验、结论、局限）及附录 A–D（密度表、域随机化参数表、Qwen3-VL 尺寸估计误差、课程式 rollout 消融）；参考文献仅浏览。本页另外 7 篇（Ego-Pi、EgoPriMo、GenHOI、VAIC、M3imic、WT-UMI、X-OP）的原文**未**阅读，其内容仅依赖 wiki 既有页面与微信周报归档。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 真机 teleop 轨迹质量高，但需要昂贵机器人、场地，且失败后需人工复位（图 2），还可能损坏硬件与物体；仿真可一键重置。
- **两阶段解耦（§1、§3.2）：** 阶段一在仿真中用 VR（PICO 4U：头显、手柄、双踝追踪器）实时遥操作，用 Isaac Sim 的 Real-Time 渲染并只记录机器人与物体的状态序列；阶段二离线重放并用 Path-Tracing 高保真渲染，随机化背景纹理、环境光强度与色温、相机外参，把一条演示扩成大量视觉不同的样本。
- **Real-to-Sim 资产（§3.2.1）：** Hunyuan3D 由参考图生成带纹理网格；Qwen3-VL 估计物理尺寸与材质类别；材质类别查表得密度 / 摩擦 / 恢复系数，均匀密度假设下算质量与惯量，并在采集时围绕预测值随机化。附录 C：5 个物体尺寸估计平均误差 0.3–3.0 cm。
- **动作接口（§3.2.2、§3.3.1）：** 操作者动作经 GMR 重定向为参考全身运动，再由 Teleopit（开源 RL 全身控制器）驱动；参考运动命令 m_t 为 67 维（沿用 TextOp 表示）；高层为 Transformer + Flow Matching 的 action-chunk 策略，输入 CLIP 文本、三视角 DINOv2 图像、最近 2 帧本体信息，预测未来 32 帧，推理用 Euler 10 步；低层输出 29 自由度身体关节 + 14 自由度手部，共 43 维。
- **训练配方（§3.3.2）：** 本体输入用「参考运动命令」而非机器人实际状态，使仿真与部署的输入分布一致、避免跟踪误差累积；课程式 rollout（P=4 段，前 20% 训练概率为 0，线性升到 0.8）。附录 D：无 rollout 平均成功率明显更低（2/10、1/10、0/10、0/10），有 rollout 为 8/10、7/10、8/10、10/10。
- **部署（§3.4）：** 29 自由度 Unitree G1 + 7 自由度三指灵巧手，头部 D435i + 腕部 D405；高层 25 Hz（RTX 4090），低层 50 Hz 执行 32 步 chunk。
- **采集效率（§4.1，表 1）：** 每任务 50 条成功轨迹，仿真 vs 真机用时 15.2/17.5、19.1/26.8、25.2/40.2、28.4/44.8 分钟，加速 1.15×–1.84×，任务越难差距越大；差距几乎全部来自复位等轨迹外开销。
- **增强消融（§4.2，表 2）：** 去掉全部随机化真机零样本成功率几乎为零（平均 0.05）；单项去除时光照影响最大（平均 0.30）；去纹理 0.43、去相机外参 0.63；每条轨迹渲染 5/10/15/20 个环境对应平均 0.38/0.58/0.75/0.83，约 15–20 时趋于饱和。
- **仿真 vs 真机数据（§4.3，图 5）：** 同样 50 条轨迹预算下，纯仿真数据真机成功率与纯真机数据相当、部分任务更高；论文归因于真机数据采集环境较固定、仿真渲染覆盖更广的视觉条件；仿真 + 真机等比混合优于单一来源。
- **局限（§6）：** 增强只随机化外观、不改轨迹（扰动全身状态易破坏平衡），运动多样性受操作者演示限制；自动生成资产的几何与物理参数对视觉复杂物体可能不准，会在接触丰富任务上放大 sim-to-real 差距。

## 对 wiki 的映射

- [`wiki/overview/loco-manip-8-papers-technology-map.md`](../../wiki/overview/loco-manip-8-papers-technology-map.md)
- [`wiki/entities/paper-loco-manip-04-oasis.md`](../../wiki/entities/paper-loco-manip-04-oasis.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2606.08548>
