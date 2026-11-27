# LabVLA: Grounding Vision-Language-Action Models in Scientific Laboratories

- **作者：** Baochang Ren, Xinjie Liu, Xi Chen, Yanshuo Liu, Chenxi Li, Daqi Gao, Zeqin Su, Jintao Xing, Zirui Xue, Rui Li, Xiangyu Zhao, Shuofei Qiao†, Minting Pan, Wangmeng Zuo, Lei Bai, Dongzhan Zhou†, Ningyu Zhang†, Huajun Chen
- **机构：** Zhejiang University（浙江大学）、Shanghai AI Laboratory（上海人工智能实验室）、Harbin Institute of Technology（哈尔滨工业大学）
- **版本 / 载体：** arXiv:2606.13578v2 [cs.CL]（2026-06-11 提交 v1，2026-06-15 修订 v2，HTML 全文 + 项目主页）
- **原文：** [arXiv 摘要](https://arxiv.org/abs/2606.13578) | [arXiv HTML v2](https://arxiv.org/html/2606.13578) | [官方项目页](https://zjunlp.github.io/LabVLA/)
- **项目页 / 代码：** 部分开源（官方 GitHub 仓库 [zjunlp/LabVLA](https://github.com/zjunlp/LabVLA) 以 MIT 协议公开了训练、微调、数据处理与 WebSocket 推理部署代码，Hugging Face [zjunlp/LabVLA](https://huggingface.co/zjunlp/LabVLA) 发布了模型权重；但仓库 README TODO 明确标注 **RoboGenesis** 数据合成引擎与 **LabEmbodied-Data** 语料尚未发布，处于 *coming soon* 状态）
- **核对范围：** 完整通读 arXiv HTML v2 全文（Abstract、§1–§8、Table 1–6、Figure 1–7 题注及 Appendix A–F），并核对官方项目页文本与 GitHub 仓库 `README.md` 及文件列表快照（2026-11-27）
- **通读：** 已完成

## 原文核对笔记

1. **核心问题定位：** 论文将科学实验室自动化建模为视觉-语言-动作（VLA）学习问题，指出数据分布（现有真实机器人数据缺少移液器、加热板、透明液体与固定实验协议流程）与跨本体（embodiment）执行是与模型架构同等重要的核心瓶颈。
2. **RoboGenesis 数据与流程引擎（§2，Table 1，Appendix E–F）：**
   - **环境构建（§2.1）：** 通过结构化商品摄影提示词（Appendix E）调用文生图 API，经 TRELLIS 2.0 重建 3D 网格、可选 MeshAnythingV2 减面并附带碰撞体与 URDF 元数据，批量构建 **LabAssetLibrary**（2,947 个带物理标注资产，项目页统计覆盖 16 大类、288 细分类）；配合 **LabTextureLibrary**（1,000+ 材质贴图）与六阶段贪心约束求解器（Appendix F Table 6），通过 10 项布局/接地/无碰撞校验并生成 **10,000** 个可仿真 USD 实验室场景；将机械臂配置（覆盖 Franka Panda、FR3、UR 系列、Piper、Rizon4、Festo、ARX X5/R5、Split ALOHA、Lift2、FR3 Duo、Ridgebase 移动底盘变体等 16 种平台）与场景、协议解耦。
   - **智能体工作流生成与域随机化（§2.2）：** 由原子技能库（操作类 `pick`/`place`/`pour`/`stir`/`shake`/`move`、仪器交互类 `press`/`pressZ`/`open`/`close` 及导航技能）组合长程实验协议；支持智能体自动生成工作流 YAML 并做离线可达性/冲突校验（>20 步复合流程采集成功率 >75%），校验通过后再施加六轴域随机化（场景布局、视觉杂物、相机位姿、同类资产替换、光照与 HDRI、空间微扰）及指令改写。
   - **LabEmbodied-Data 导出（§2.3）：** 仅导出通过单步物理成功判据与接触安全监控（拒绝违禁碰撞）的成功轨迹，附带 15 类可配置结构化标注流（机器人状态、相机内外参、步骤起止帧、子指令对齐、物体状态码、场景空间关系、物体语义、成功判据解释、碰撞事件、时序分段、子目标帧、质量评分、人工干预标志与回合元数据）。
3. **LabVLA 训练配方（§3，Appendix A–C）：**
   - **架构：** 以 **Qwen3-VL-4B-Instruct**（$d_{\mathrm{vlm}}=2560$）编码多视角图像（$224\times 224$）与语言指令，通过线性投影 $\Pi(H_\phi)$ 条件化一个独立 **18 层 DiT 动作专家**（宽度 1024、8 注意力头、每头 128 维），输出 $K=50$ 步连续动作块；所有本体状态/动作统一填充至 $d_{\max}=32$ 并通过有效掩码 $M^{\mathrm{act}}$ 屏蔽无效维度与帧。
   - **阶段 1（VLM FAST 预训练，§3.1）：** 在 Robointer-VQA、AgiBot World Beta、OXE-AugE（LeRobot 子集约 572k 轨迹）与 DROID 上用离散 FAST 动作 token + VQA/标注交叉熵训练 VLM（24×A100，全局 batch 1536，100k 步），状态离散为 256 个均匀分箱写入文本前缀。
   - **阶段 2（流匹配后训练 + 知识隔离，§3.2–3.3）：** 联合 OXE-AugE 与 LabEmbodied-Data 接入 DiT，采用整流/流匹配损失 $\mathcal{L}_{\mathrm{FM}}$（$\tilde{\tau}\sim\mathrm{Beta}(1.0,1.5)$，$\tau=0.999\tilde{\tau}$），并用 stop-gradient $\widetilde{H}_{\phi,p}^{\mathrm{KI}}=\mathrm{sg}(\mathrm{slice}_p(f_\phi(X_{\mathrm{KI}})))$ 阻止流匹配梯度回传破坏 VLM 语言与视觉锚定能力，联合损失 $\mathcal{L}_{\mathrm{KI}}=\alpha\mathcal{L}_{\mathrm{FM}}+\mathcal{L}_{\mathrm{FAST}}+\sum_j\lambda_j\mathcal{L}_{\mathrm{CE}}^{(j)}$（$\alpha=10$；16×A100，全局 batch 1024，80k 步）。推理时仅需 $N=10$ 步欧拉积分。
   - **训练排障与系统优化（Appendix B–C）：** 修复了动作填充维度参与均值导致 8-DoF 单臂梯度被隐式缩小 $4\times$ 的问题；启用仅 LM 子模块选择性梯度检查点、Liger-Kernel 融合算子（含 FLCE 节省约 10 GB fp32 logits 显存）、FlashAttention-2 1D 因果掩码（规避 4D 分块掩码带来的约 30% 步时开销）、CUDA 独立传输流预取以及共享 PyAV LRU 缓存（上限 256）+ `MALLOC_ARENA_MAX=2` + `malloc_trim(0)`（worker RSS 均值下降 55%）。
4. **实验结果（§4–5，Table 2–4）：**
   - **LabUtopia 基准（Table 2，每设定 120 回合）：** LabVLA 平均成功率在 ID 达 **71.1%**、OOD 达 **70.0%**（仅下降 1.1 pp），高于次优基线 $\pi_0$（ID 63.3% / OOD 63.2%）及其他基线（X-VLA、GR00T N1.5、$\pi_{0.5}$、SmolVLA、InternVLA-A1、$\pi_0$-FAST、Wall-oss-flow）；倒液（Pour Liquid）对所有模型均最困难（LabVLA ID 43.3% / OOD 34.2%）。
   - **数据可迁移性（Table 3）：** 将 LabEmbodied-Data 用于微调外部模型 X-VLA（<1B），5 项非饱和任务平均成功率在 ID 提升 +15.0 pp（49.3% → 64.3%），OOD 提升 +19.3 pp（43.7% → 63.0%）。
   - **真机 Franka 评测（Table 4，每设定 50 回合）：** 在 Shake Liquid、Pour Liquid、Magnetic Stir、Funnel Plug/Unplug 四项多步操作上，仿真训练迁移至真机取得 ID Clean 86.5%、ID Cluttered 80.0%、OOD Clean 80.0%、OOD Cluttered 74.0%，整体与 DreamZero 接近且优于 $\pi_{0.5}$。
   - **能力分层定位（§8，Figure 6）：** 作者明确将 LabVLA 定位于实验室具身能力四级金字塔的 **Level 2（Technician，按固定多步协议执行）**，尚未达到精密仪器与安全测量闭环的 Level 3（Specialist）或自主修改实验方案的 Level 4（Scientist）。
5. **对 wiki 的映射：** [LabVLA 论文实体页](../../wiki/entities/paper-agentic-2606-13578.md)
