# 4DCodeBench: Benchmarking Agents on Inverse Graphics of Dynamic Scenes

- **作者：** Ruihong Shen, Žiga Kovačič, Peter Kulits, Xingrui Wang, Zizhang Li, Joshua B. Tenenbaum, Alan Yuille, Jieneng Chen, Jiajun Wu
- **机构：** Johns Hopkins University, Stanford University, Max Planck Institute for Intelligent Systems, Massachusetts Institute of Technology
- **版本 / 载体：** arXiv:2610.03715v1 [cs.CV]（2026-10-02 提交；HTML 全文版 + 官方项目页）
- **原文：** https://arxiv.org/html/2610.03715 （摘要页：https://arxiv.org/abs/2610.03715 ；项目页：https://4dcodebench.com/ ）
- **项目页 / 代码：** **宣称将开源 / 未确认官方仓库可运行源码**。项目页（https://4dcodebench.com/ ，HTTP 200）正常可访问，包含排行榜、案例可视化及 README 摘要，提及官方仓库 `https://github.com/4DCodeBench/4DCodeBench`（含 `scripts/download_data.py`、`harness/runtime/infer.py`、`harness/runtime/score.py`、`vlm_judge/judge.py`）与 Hugging Face 数据集 `4DCodeBench/Results`；但截至 2026-11-27 缓存核查（`candidate_repos`）未收录并确认该官方仓库的真实代码文件树，仅核实到其调用的外部依赖仓 `ArtificialAnalysis/Stirrup`（开源模型 agent 框架）、`InteractiveComputerGraphics/SPlisHSPlasH`（SPH 流体仿真器）与 `NVIDIA/warp`（GPU 仿真库），因此暂按未确认官方可运行源码处理，不创建 `sources/repos/` 归档。
- **核对范围：** 通读 `arXiv:2610.03715v1` 全文（Abstract、§1–§6 及 Appendix A–G，含 Eq. 1–11、Table A1–A5、Figure 1–8 与 A1–A22 说明）及项目页 `https://4dcodebench.com/` 全文文本（Overview、Key takeaways、Discussion、README 节选与 FAQ）。
- **通读：** 已完成

## 原文核对笔记

对 wiki 的映射：[4DCodeBench：面向动态场景逆向图形学的代码智能体基准](../../wiki/entities/paper-agentic-2610-03715.md)

### 1. 任务形式化与提交协议（§2.1, Appendix G.1–G.3）
- 输入仅包含参考视频 $x_{\mathrm{ref}} = (x_0, \dots, x_{T-1})$ 与固定任务说明（`TASK.md`），不提供场景名称、物体列表或物理参数；模型运行在单 GPU、6 小时超时、挂载只读输入和空工作区的隔离容器中（Ubuntu 24.04 + CUDA 13.0 + Blender 5.2.0 + NumPy/SciPy/PyTorch/Taichi 1.7.4/Warp 1.17.0/Trimesh/Pillow 及离线 Blender API 文档）。
- 智能体生成可执行程序 `/workspace/solution/build.sh` 及其 Python 代码，在不读取 $x_{\mathrm{ref}}$、不联网、不自带外部资产的前提下从零重建 `/workspace/world/`，包含固定针孔相机 `camera.json`（$3\times 3$ 内参 $K$ 与 $4\times 4$ 外参 $E$）、Blender 渲染视频 `render.mp4`、逐帧网格 `meshes/<id>.npz`、持续粒子/顶点轨迹 `dynamics/<name>.npz`（或无粒子身份欧拉流体烘焙 `solver/` + `manifest.json`）。
- 禁止复制参考视频像素作为贴图；允许自由选择动画、Blender 物理或 Taichi/Warp/自定义求解器实现动力学。

### 2. 数据集构成（§2.2–§2.3, Appendix C）
- 共 **200 个动态场景**：**100 段真实视频** + **100 个合成场景**（带完整 4D 真值），帧率上限 60 fps，真实视频帧数上限 300 帧；优先静态机位、连续无剪辑，排除人体与动物并尽量减少手部出镜。
- 四大物质类别：刚体与铰接体（rigid & articulated）、可变形固体（deformable solids）、余维结构（co-dimensional：布料/绳索/壳/杆）、流动物质（flowing materials：颗粒/牛顿流体/黏塑性流体）。83% 含多动态对象，66% 含多物质耦合；区分被动动力学与外部驱动（含机器人操作）。
- 真实子集（Table A2）：物理视频基准 51 段（WISA-80K 33、Physics-IQ 12、Phys-AD 3、Phys101 3）、机器人操作数据集 21 段（ABC-130K 5、T-Rex 4、Robo360 3、AgiBot World 2、RoboCook 2、RoboCraft 2、ALOHA Unleashed 1、ManipArena 1、Task-Level ILC 1）、网络采集 28 段（Pexels 21、YouTube CC 5、Mixkit 1、Pixabay 1）。
- 合成子集（Table A3）：Blender 31、Genesis 12（作者新增黏塑性流体、Stomakhin 雪模型、Drucker–Prager 沙模型三种 MPM 本构）、libuipc 10、CK-MPM 9、PPF 9、HOT 7、MPM-lite 4、AnisoMPM 4、CD-MPM 4、SPlisHSPlasH 3、MLS-MPM 3、Silly Rubber 3、IQ-MPM 1。

### 3. 评测指标与人类对齐（§2.4, §3.2, Appendix D–E）
- 五大指标族等权平均为 Overall（失败项计最差惩罚值 0 或深度误差 2）：
  1. **Perceptual**：全场景 DINOv3 class-token 余弦相似度映射至 $[0,1]$。
  2. **2D Dynamics**：全场景 Dynamic IoU；真实视频叠加 `Flow`（解析表面流 vs. RAFT 的归一化切片 Wasserstein 距离）与 `Track2D`（CoTracker3 轨迹 DTW，$\kappa=0.03$）。
  3. **2.5D Geometry**：真实视频 `Depth error`（栅格化网格视差 vs. Video Depth Anything，中位数/MAD 归一化并剔除最大 1% 残差，上界 2）。
  4. **3D Geometry**：合成场景对齐（trimmed ICP 相似变换）后第 0 帧对称 Chamfer 距离转化的 `Scene 3D`（$\kappa=0.5\sigma$）。
  5. **3D Dynamics**：合成场景拉格朗日视角 `Trajectory DTW`（Hungarian 匹配，$\kappa=1$）与欧拉视角 `EMD step`（单步 3D 位移切片 Wasserstein 距离）。
- **VLM VQA & 成对 Elo**：1,251 道人工编写二元题（初始状态 371、末态 354、关键事件 308、接触 218），由 Gemini 3.8 Flash 看 16 帧评测（含空/纯色屏与主体存在性双重门控，81.0% 正常进入裁判）；成对盲测通过 Bradley–Terry 模型拟合 VLM Elo。
- **人类研究**：76 名参与者对 17 个模型（不含因发布时间未入场的 Opus 5.5 [High]）完成 3,587 次判断；Human Elo 与 VLM Elo Spearman $\rho=0.980$，与 Overall 分数 $\rho=0.96$；916 组共享比较中人机一致率 89.3%（$\kappa=0.761$），接近人类内部一致率 92.1%（$\kappa=0.816$）。

### 4. 主要实验结论与消融（§4, Appendix F, Appendix G.4）
- 评测 18 个多模态编程模型（含专有 10 配置与开源权重 8 模型，开源模型通过 Stirrup + SGLang 运行），共 3,600 次运行、5,983 小时、$16,495 算力开销。
- 总体可执行率 **90.1%**（专有模型均 $\ge 96.5\%$，开源模型 29.0%–95.5%）。`GPT-6 Astra [Max]` 以 Overall **0.791** 领跑，其次为 `Claude Opus 5.5 [High]`、`GPT-6 Astra [High]`（0.77）、`Claude Fable 5.1 [High]`、`GPT-6 Astra [Low]`（0.726）。
- **静动态落差显著**：所有模型的外观/静态几何均明显高于动力学重建（如 `Astra [Max]` 静态族均值 **0.91** vs. 动态族均值 **0.67**）。
- **动力学建模策略分化**：汇总全部提交中 **67%** 采用解析运动（样条、顶点位移映射、弹道公式等）、**19%** 自写仿真（Taichi 155 次、Warp 102 次；PBD/XPBD 136、MPM/MLS-MPM 111 等）、**10%** Blender 物理、**3%** 关键帧。`Opus 5.5 [High]` 自定义仿真达 **61%**（含 Blender 共 76%），而 `Astra [Max]` 解析运动占 63%、`Terra [High]` 达 100%。
- **推理预算效应**：跨模型间步数与 token 总量与 Overall 相关性极弱（$|\rho| \le 0.10$），但固定模型（`GPT-6 Astra` Low → High → Max）提高推理强度可将 Overall 从 **0.726** 提至 **0.77** 再到 **0.791**（2D Dynamics 0.48→0.60，3D Dynamics 0.63→0.73，VQA 78.2%→87.6%）。
- **主要局限**：仅评测视觉/几何重现保真度而未做反事实初始条件/外力干预或长时外推验证，使得解析拟合曲线也能取得高分；且仅评估终稿而非 render-and-compare 中间迭代过程。
