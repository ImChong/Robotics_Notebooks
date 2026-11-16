# Qwen-RobotManip Technical Report: Alignment Unlocks Scale for Robotic Manipulation Foundation Models

- **作者：** Qwen Team（核心贡献者：Haoqi Yuan, Zhixuan Liang, Anzhe Chen, Ye Wang, Haoyang Li, Pei Lin, Yiyang Huang, Zixing Lei, Tong Zhang 等；通讯/指导：Chenfei Wu, Xiong-Hui Chen）
- **链接：** https://arxiv.org/abs/2606.17846 （注：本归档沿用集合期索引文件名 `2610_41880`，正文与仓库 BibTeX 核验正式 arXiv 编号为 `2606.17846`，v1 发布于 2026 年 6 月 16 日；官方页 https://qwenlm.github.io/Qwen-RobotManip/ ）
- **状态：** primary_full_text_checked
- **核对范围：** 已通读 `arXiv:2606.17846` 技术报告全文（1–4785 行：总览、`Qwen3.5-4B` + Flow-Matching DiT 架构、80D 统一状态动作空间、相机系相对 EEF + CaPE 空间对齐、~38,100 小时多源及 H2R 重定向数据管线、Unified-mode In-Context Policy Adaptation、LIBERO / RoboTwin / RoboCasa365 / EBench / RoboChallenge 与 ALOHA / ARX 实机评测及消融实验）
- **通读：** 已完成

## 核心要点提取

1. **统一状态动作表示与相机系三维几何对齐**
   - 定义 **80 维** 统一连续状态/动作向量：`2 × 29D` 单臂槽位（7 关节 + 9 EEF [状态为 3D 位置 + 6D 旋转，动作为 3D 平移 + 3D 旋转向量] + 1 夹爪 + 12 灵巧手自由度）+ `22` 预留维；通过每样本二值掩码 `m_{i,t,j}` 对有效自由度做激活掩码与归一化损失（Eq. 11）。
   - **Camera-frame Delta EEF + CaPE**：将末端增量统一变换至当前相机坐标系 `Eq.(5)`（消除世界原点任意性与跨机台安装标定差异），并在动作专家注意力中引入 32/64 维头部相机位姿编码（CaPE）与 32 维 RoPE，同时用 `AdaZeroRMSNorm` 注入 `eef_type` 与 `has_camera_params` 调制。
   - **结构化具身提示（Structured Embodiment Prompt）**：显式输入本体构造、夹爪形态与控制帧率（训练期 15% 随机丢弃以防死记），支持 Unified-mode In-Context Policy Adaptation（随机块采样 `(o_h, s_h, a_h)` 防止仅复制末尾动作的退化捷径）。

2. **“对齐解锁规模”：~38,100 小时全开源跨载体与 H2R 数据管线**
   - 纯用开源数据，无闭源内部遥操采集：
     - **真实与仿真机器人轨迹（11,420 小时）**：单臂 3,808h、双臂 6,744h、移动/人形 868h（含 OXE ~600h、AgiBotWorld-Beta ~2,400h、RoboMIND 1.0/2.0 ~1,400h、Galaxea ~500h、RoboCOIN ~430h、DROID ~500h、RH20T ~1,100h、RDT-1B 29h、InternData-A1 >3,600h 仿真等）。
     - **第一视角人手操作视频（1,933 小时）**：EgoDex 732h、VITRA 247h、EgoVerse 954h 转 MANO 手部轨迹。
     - **H2R（Human-to-Robot）合成重定向数据（24,808 小时）**：覆盖 15 种双臂机械臂构型（SAM3 分割 + ProPainter 修复背景 + Savitzky-Golay / 高斯 SLERP 平滑 + Eq. 3 基座网格搜索优化 + MuJoCo `mink` IK 求解 + Depth Anything v3 遮挡合成 + 分数据集速度降采样）。
   - **五阶段运动学清洗 + 三项跨模态核验**：基于突变检测、趋势相关性（剔除 RoboMIND UR 数据 81%）、`[q01, q99]` 极值截断、Pinocchio FK 一致性校正（校准 TCP 与参考系）、基座 +x 朝向对齐；结合 28M VL 共训混合语料以 9:1（VLA:VL）比例联合优化 `L = L_FM + 0.1 * L_VLM`。

3. **主干与关键实验结果**
   - 架构为 `Qwen3.5-4B` VLM（`D_vlm=2560`）+ `N=10` 层 Flow-Matching DiT（`D_act=768`，12 头，偶层视觉/奇层语言末层交叉注意力，可学习 Register/Query tokens 优于逐层 KV 缓存交互；训练 `t ~ Beta(1, 1.5)`，`K_repeat=8`，标准推理 4 步 Euler，`Context` 变体 10 步配 RTC 远程推理）。
   - **仿真 SOTA**：LIBERO `99.1`（Context `99.2`）、RoboTwin Easy/Hard `93.4 / 92.5`（Context `93.7 / 94.0`）、LIBERO-Plus `89.0 / 91.4`（对比 $\pi_{0.5}$ `84.4`）、RoboTwin-C2R Hard `62.6 / 69.4`（对比 $\pi_{0.5}$ `47.9`）、RoboCasa365 `35.9` 总分、EBench `45.6` SR / `60` 分、RoboTwin-IF `72.2`（对比 $\pi_{0.5}$ `49.6`）、RoboTwin-XE 跨构型零样本 `23.9`（对比 $\pi_{0.5}$ `7.5`）。
   - **真机评测**：CobotMagic ALOHA ID `88.6%` / OOD `87.5%`；ARX 跨载体零/少样本迁移 `55.0%`（去掉 UnifiedEEF 骤降至 `12.5%`，去掉 UnifiedSpace 降至 `7.5%`）；RoboChallenge Table30-v1 全榜单第一（`Lira_generalist` `45%` SR / `59.83` 分，超越 DM0 `37%` 与 $\pi_{0.5}$ `17.67%`）。

## 局限性与边界条件

- 模型权重与训练/推理源码均未公开（仅释放技术报告与展示页）。
- In-Context 版本因开局零填充历史块会出现轻微起步迟疑；在高频力控/精细插装上仍受 8–10 Hz chunk 延迟与无触觉模态输入限制。
