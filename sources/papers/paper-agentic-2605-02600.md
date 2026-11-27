# CoRAL: Contact-Rich Adaptive LLM-based Control for Robotic Manipulation

- **作者：** Berk Çiçek, Mert K. Er, Ozgur S. Oguz
- **机构：** LiRA Lab, Department of Computer Engineering, Bilkent University (Ankara, Türkiye)
- **版本 / 载体：** arXiv:2605.02600v2 [cs.RO]（2026-05-12，v1 提交于 2026-05-04；RSS 2026 接收，含附录 S1–S12 共 22 页）
- **原文：** [arXiv 摘要页](https://arxiv.org/abs/2605.02600) · [arXiv HTML 全文](https://arxiv.org/html/2605.02600)
- **项目页 / 代码：** [项目主页](https://sites.google.com/view/lira-coral) · **未确认开源代码**（项目主页仅展示摘要、任务演示视频 `CoRAL_Summary_6.mp4` 及仿真/真机结果对比；论文正文与附录 S7–S12 给出部分 Python 伪代码与提示词模板，但未提供官方代码仓库链接；候选抓取中出现的 `google/safevalues` 为 Google Sites 网页依赖库，非本论文实现）
- **核对范围：** 已通读 arXiv:2605.02600v2 HTML 全文（摘要、I 引言、II 相关工作、III-A 至 III-D 方法论、IV 实验设计与 Table I–III 全部结果、V 局限与结论，以及附录 S1–S12 全部数学推导、Table S1、消融与 Prompt/代码清单）及项目主页文本。
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 端到端 VLA（如 OpenVLA-OFT、$\pi_{0.5}$）依赖大规模示教数据，在需要主动利用外部接触和力调节的任务上泛化脆弱；而已有基于 LLM 生成代价/奖励的规划方法（如 L2R）依赖静态物理假设，缺乏在线系统辨识与策略修正机制。
- **架构设计：** 将高层次语义推理与低频/高频控制解耦，包含三层控制回路：
  1. **感知与世界模型初始化（§III-A）：** 仿真中使用 FoundationPose（RGB-D + 3D 物体模型 $M$）持续跟踪 6-DoF 位姿，真机采用 6 相机 Vicon Vero 动捕替代；GPT-4o VLM 根据图像、任务描述与物体标签给出质量与摩擦系数等物理参数语义先验 $\theta$（JSON 格式）。
  2. **任务建模与经验记忆检索（§III-B，§S12）：** 先通过基于 LLM 语义嵌入的 RAG 记忆单元检索历史成功经验 $(J_{\text{mem}}, C_{\text{mem}})$；若未命中，由 LLM（Task Formulation）生成初始 Python 运行代价函数 $J_0$（含多阶段软 sigmoid 切换）以及接触区域描述符（center、normal、extent、num_samples），再经 `sample_points_in_region` 在局域流形 $\mathcal{R}=\partial\mathcal{O}\cap\mathcal{E}(c_{\text{ref}},\mathbf{A})$ 上采样形成软吸引子接触策略 $C_0$。
  3. **双嵌套闭环执行（§III-C，§III-D）：**
     - **内环（高频反应式控制）：** 1 kHz 关节阻抗控制（Tier 1）+ 10 Hz MuJoCo/robosuite MPPI 滚动优化（Tier 2，$K=256, H=32$，基于 ESS 目标 $\phi=0.2$ 自适应调节温度 $\lambda$），输出叠加传感反馈增强项 $\nu_t = u_t + K_f (x_{\text{des}} - x_{\text{measured}})$。
     - **外环（低频在线自适应）：** 连续失败 $N_{\text{retry}}$ 次后触发 LLM（Online Adaptation，约 1 Hz 异步，Tier 3），基于执行日志 $E_t$ 执行两类修正：（a）世界模型校正（在线系统辨识，更新质量/摩擦参数 $\theta$）；（b）策略与代价函数重写（修改 `state_cost` 源码并输出自然语言故障诊断）。
- **关键实验数据（§IV，§S3）：**
  - 在 6 项仿真任务（含 4 项接触与力关键任务 T1/T4/T5/T6 与 2 项标准抓取放置 T2/T3，每项 10 次随机化测试）中，CoRAL 总体成功率显著领先 OpenVLA-OFT、$\pi_{0.5}$ 与 L2R（例如 T1 5/10、T4 9/10、T5 9/10、T6 7/10，而基线在 T1/T6 几乎为 0–1/10）。
  - 敏感性与误差分析（Table II）：T1 任务中 VLM 初值质量/摩擦 MAE 为 0.29 kg / 0.14，经 4 轮外环自适应后降至 0.11 kg / 0.06（成功率从 2/10 升至 5/10）；若改用真值位姿则升至 7/10，表明接触遮挡下的位姿跟踪误差（FoundationPose 均值 ADD 13.2 mm，88.2% 帧 < 5 mm）是主要瓶颈。
  - 真机零样本部署（Table III，Franka Emika Panda + Vicon Vero）：T1 4/10、T2 10/10、T3 10/10、T4 9/10（力控稳定在约 5 N）、T5 7/10、T6 6/10。
- **对 wiki 的映射：** [CoRAL 论文实体页](../../wiki/entities/paper-agentic-2605-02600.md)
