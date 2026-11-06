# RoboTwin 2.0 论文通读（双臂数据生成与基准视角）

> 来源归档（原文通读 · 支撑 `wiki/entities/robotwin.md`）

- **标题：** RoboTwin 2.0: A Scalable Data Generator and Benchmark with Strong Domain Randomization for Robust Bimanual Robotic Manipulation
- **作者：** Tianxing Chen, Zanxin Chen, Baijun Chen, Zijian Cai, Yibin Liu, Zixuan Li 等（SJTU / HKU MMLab / Shanghai AI Lab 等；通讯 Ping Luo、Yao Mu）
- **类型：** paper / simulation / data-generation / bimanual
- **原文：** <https://arxiv.org/abs/2506.18088>（2026-11 通过 arXiv PDF v2 获取全文，pdftotext 提取）
- **一句话说明：** 用 MLLM 代码生成 + 仿真闭环反馈自动合成双臂专家轨迹，配 RoboTwin-OD 资产库、五轴域随机化与跨本体抓取适配，并给出 50 任务基准与 sim-to-real 验证。
- **核对范围：** 通读摘要、§1 引言、§2 方法（2.1–2.3）、§3 数据集与任务、§4 实验（4.1–4.5，表 1–5）、§5 相关工作、§6 结论，以及附录 A–H、附录 J–L 的文字与表格；附录 I（指令/描述生成的 prompt 全文）和附录 G.5（代码案例）、附录 K/L 的 50 任务逐项数值仅浏览，未逐项核对。未读官方文档站与 GitHub 代码。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **问题（§1）：** 现有合成数据缺自动质量控制、域随机化流于表面、忽略跨本体差异（如低自由度 Piper 倾向侧抓，Franka 可顶抓）。
- **专家代码生成（§2.1，图 3）：** 双 agent 闭环——代码生成 agent（DeepSeek-V3，附录 G）依据任务名、API 列表、调用示例与层级约束生成 Python 程序；每轮在仿真中跑 10 次并写执行日志（不可执行、左/右抓取失败、放置位置错误等）；VLM 观察 agent（moonshot-v1-32k-vision-preview）逐帧定位失败步骤并诊断原因；两路反馈驱动代码修复；单轮 10 次成功率达设定阈值（图中为 >0.5）即通过，连续 5 次修复仍不达标则放弃。
- **域随机化五轴（§2.2）：** 杂物干扰物（取自 RoboTwin-OD，碰撞感知放置，排除与目标相似者）、背景纹理（1000 条描述 × Stable Diffusion v2 各 20 张 → 人工筛得 11000 张）、光照（颜色/类型/强度/位置）、桌面高度（附录 C：最多 3 cm）、轨迹级语言指令（MLLM 生成模板与物体描述后组合采样）。
- **本体感知抓取适配（§2.3）：** 每个物体标注多抓取轴/接近方向的候选操作位姿，加偏向高可达性的角度扰动，并行运动规划（附录 E：cuRobo）；支持 Franka、Piper、UR5、ARX-X5、Aloha-AgileX 及任意双臂组合。
- **RoboTwin-OD（§3.1）：** 731 实例 / 147 类；534 个 / 111 类自建（Rodin RGB→3D + 凸分解），153 个来自 Objaverse（27 类），44 个铰接体来自 SAPIEN PartNet-Mobility（9 类）；每物体 15 条语言描述，并标注放置点、功能点、抓取点、抓取轴。
- **任务与数据（§3.2）：** 50 个双臂任务、5 种本体，预采 10 万+ 轨迹（HuggingFace 发布）。
- **代码生成评测（§4.1，表 1）：** ASR：R1.0 Vanilla 47.4% → R1.0 多模态反馈 63.9%；R2.0 Vanilla 62.1% → R2.0 + 多模态反馈 71.3%；修复迭代 1.76 vs 2.42；Vanilla 代码 token 569.4 vs 1236.6（附录 F）。VLM 观察器仅在失败时触发，平均约 6894 token，可关闭。
- **跨本体抓取（§4.2，表 2）：** 50 任务自动采集成功率，平均 52.2% → 60.5%（+8.3%）；Piper 2.4% → 25.1%，Aloha-AgileX +13.7，ARX-X5 +5.6；Franka、UR5 基本持平。
- **鲁棒性（§4.3，表 3）：** 在 32 任务各 300 条（共 9600）上预训练 RDT/Pi0；干净数据预训练对 5 个未见任务几乎无提升，域随机化预训练使 RDT、Pi0 相对提升 31.9%、29.3%（Pi0 22.5 → 29.1，平均）；下游微调仅用干净数据。
- **Sim-to-real（§4.4，表 4）：** COBOT-Magic 平台、RDT、4 个任务；10 条真机 + 1000 条随机化合成，平均成功率提升 24.4 个点（各配置 +13.5/+27.5/+23.5/+33.0）；纯合成零样本在两个未见背景配置上 +21.0/+20.5；仿真相机位姿加 ≤1 cm 随机扰动。
- **基准（§4.5，表 5）：** Aloha-AgileX 上 50 任务、50 条干净演示训练、各 100 次 rollout；Easy/Hard 平均：RDT 34.5/13.7，Pi0 46.4/16.3，ACT 29.7/1.7，DP 28.0/0.6，DP3 55.2/5.0。非预训练模型 Hard 近乎崩溃；预训练 VLA 较稳但 Easy→Hard 仍降 20.8（RDT）/30.1（Pi0）个点；DP3 的好成绩部分来自仿真里完美点云与干净分割。
- **结论与未来（§6）：** 后续重点是真机部署与多物体任务复杂度。

## 对 wiki 的映射

- [`wiki/entities/robotwin.md`](../../wiki/entities/robotwin.md)

## 参考来源

- 论文：<https://arxiv.org/abs/2506.18088>
