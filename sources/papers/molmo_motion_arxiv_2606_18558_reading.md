# MolmoMotion 论文原文精读归档（arXiv:2606.18558v2）

- **类型：** 论文精读
- **标题：** *MolmoMotion: Forecasting Point Trajectories in 3D with Language Instruction*
- **作者与机构：** Jianing Zhang, Chenhao Zheng, Yajun Yang, Max Argus, Rustin Soraki, Winson Han, Taira Anderson, Chun-Liang Li, Shuo Liu, Jiafei Duan, Zhongzheng Ren, Jieyu Zhang, Ranjay Krishna（Allen Institute for AI / University of Washington / UNC-Chapel Hill）
- **版本：** arXiv:2606.18558v2（2026-10-19）
- **原文链接：** <https://arxiv.org/abs/2606.18558>
- **通读：** 已完成
- **核对范围：** 论文全文第 1–7 节及附录 B–F，重点核对任务形式化（目标条件 metric 3D 点轨迹预测）、`MolmoMotion-1M` 自动标注管线与 7 大语料统计（Table 3）、`PointMotionBench` 742 条人工校验基准构成（Table 4）、主实验对比（Table 1）、设计消融（Table 6）、AR 与 FM 推理时延对比（Table 7）、MolmoSpaces / DROID 机器人迁移（Sec. 4.2 + Appendix F）与 DaS + CogVideoX-5B 轨迹引导视频生成（Sec. 4.3 + Table 2）。

---

## 一、核心方法与模型变体

1. **任务定义与锚点相对参数化**
   - 给定历史 `H` 帧 RGB（`H = 1` 或 `3`）、`t_0` 时刻物体上的 `N` 个 2D query 点及其初始 3D 位置与自然语言动作描述，预测未来 `T` 帧（15 fps，最长约 2 秒 `T = 30/32`）的相机/世界坐标系 3D 点轨迹（单位：米）。
   - 采用 **锚点相对坐标增量（anchor-relative delta）** 并按 1 mm 粒度量化（`×1000`），配合骨干视觉编码器在 query 2D 坐标处网格采样的局部视觉特征，解决绝对世界坐标分布跨度大、单靠文本难以对齐几何初值的问题。
2. **Molmo2-4B 骨干与两阶段训练（Sec. 4.1）**
   - 基于预训练 VLM **Molmo2-4B**（Qwen3-4B LM + SigLIP2/DINOv2 视觉编码器）。
   - Stage 1（`P=8, H=3, F=8`，40K steps）：5 个开源人类视频语料按 `√N_i` 混合采样，学习短时程 3D 动力学。
   - Stage 2（10K steps）：将预测时域扩展至 `T=32`，分别训出 `H=3` 与 `H=1` 两个版本以兼顾短时历史与单帧关键帧输入。
3. **自回归（MolmoMotion-AR）vs 流匹配（MolmoMotion-FM）**
   - **MolmoMotion-AR**：将 `<tracks coords="...">` 作为文本 token 自回归生成，严格从左到右条件化，在 PointMotionBench 确定性轨迹精度指标上表现最优。
   - **MolmoMotion-FM（Appendix D/E）**：在 Molmo2 LM 每层隐状态之上附加 36 层 DiT cross-attention 速度头 `v_φ`，以 `K=10` 步欧拉积分从高斯噪声生成连续轨迹，可建模多模态不确定性未来。

---

## 二、数据集与基准构成核对

1. **MolmoMotion-1M（1.16M 源视频 → 约 1M 运动片段，736 动词，5,692 物体类，每片段中位 88 个 query 点，Table 3）**：
   - EgoDex（~160K，人类第一视角）、HD-EPIC（~21K，厨房第一视角）、Xperience（~500K，人类第三人称）、MolmoSpaces（~185K，仿真）、DROID（~27K，真机遥操作）、YT-VIS（~2K，自然场景）、Stereo4D（~70K，立体深度户外/可变主体）。其中公开基础训练配方仅使用 EgoDex / HD-EPIC / Xperience / YT-VIS / Stereo4D 五个来源。
   - 标注流水线：Molmo2-8B 动作重述 + Qwen3-0.6B 提取物体名 → MolmoPoint-8B 动作提示定位 → SAM 3 掩码 + K-means 采 `N=100` 点 → AllTracker 2D 追踪 + ViPE 单目度量深度/位姿提升至 3D → 16 个稳定锚点一致性滤波与运动窗口切分。
2. **PointMotionBench（总计 742 条人工校验片段，111 类物体，61 类动作，Table 4）**：
   - HOT3D（497 clips，网格真值）、WorldTrack（155 clips，动捕点云）、DAVIS（90 clips，人工编写动作与轨迹校验）。

---

## 三、关键实验、消融与时延结论

1. **PointMotionBench（Table 1，best-of-5）**：
   - `MolmoMotion-AR (H=3)` 在 HOT3D（`ADE 0.109 / FDE 0.217 / PWT 0.444`）与 WorldTrack（`ADE 0.143 / FDE 0.261 / PWT 0.445`）均大幅领先像素视频预测基线 `Wan2.2-5B`（`0.200 / 0.852` ADE）和 `Cosmos Predict`（`0.225 / 0.831` ADE）以及 2D 提升基线 `Track2Act`（`0.294 / 1.230 / 4.853` ADE）；`H=1` 在包含复杂摄像机运动的 DAVIS 上达到最佳 `ADE 1.146 / PWT 0.199`。
2. **设计消融（Table 6）**：去掉锚点相对参数化会导致各数据集 ADE/FDE 下降约 50%（影响最大）；移除语言指令导致同等量级退化；去掉 2D 网格点特征带来 5–8% ADE/FDE 退化；将 `N=8` 加倍至 `N=16` 虽带来 2–3% 微小提升，但使自回归回答跨度超过 Molmo2（Qwen3-4B）`4096` 上下文窗口，因此论文 Stage 2 默认保留 `N=8`。
3. **推理速度权衡（Table 7，单卡 A100，H=3）**：`MolmoMotion-AR`（`N=8, T=32`）耗时 **`148.4 s/clip`**（`T=8` 时为 `37.2 s/clip`），而 `MolmoMotion-FM`（`K=10` 步）仅 **`1.1 s/clip`**（快约 **150 倍**，以轻微 ADE 损失换取闭环实时性）。
4. **下游迁移（Sec. 4.2–4.3 & Table 2）**：
   - **机器人操作（MolmoSpaces Pick-and-Place & DROID）**：以 MolmoMotion 初始化 `MolmoBot` 骨干，10K 步即达 **51%** 成功率（Molmo2 初始化仅 **19%**），最终平均成功率 **76.3% vs 56.0%**；DROID 实机单目视频微调也呈现显著更低的轨迹 L2 误差与更快收敛（实机闭环尚待未来验证）。
   - **视频生成（DaS + CogVideoX-5B）**：`DaS + MolmoMotion` 在 PointMotionBench VBench 五项指标全面高于无引导 `CogVideoX-5B`，并在 `Tem-Con 0.968`、`Subj-Cons 0.950`、`M-Smooth 0.990`、`Bg-Cons 0.948` 四项上优于更大参数量的 `Wan-14B`。
