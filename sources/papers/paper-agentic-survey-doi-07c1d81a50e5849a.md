# Adaptive Task Planning for Long-Horizon Robotic Manipulation Based on Video Priors and Dynamic Scene Graphs

- **作者：** Guanghui Ma, Jiahui Guo, Xinhua Tang, Huaidong Zhou, Yongfeng Rong
- **机构：** Anhui Polytechnic University（安徽工程大学）；Northwestern Polytechnical University（西北工业大学）；Shenzhen University（深圳大学）；Tsinghua University（清华大学）（注：通讯作者邮箱为 `hdzhou@buaa.edu.cn`）
- **版本 / 载体：** *Sensors* (Basel), 2026, 26(17), 5595；DOI: `10.3390/s26175595`；PMCID: `PMC13568281`；PMID: `42740215`（Received 2026-07-20；Revised 2026-08-28；Accepted 2026-08-30；Published 2026-09-03）
- **原文：** [MDPI Sensors 原文入口](https://www.mdpi.com/1424-8220/26/17/5595)（全文核对文本来源：PMC 全文归档 `PMC13568281`）
- **项目页 / 代码：** 未确认开源代码（截至 2026-11-27 检索与原文核查：正文与 Data Availability Statement 均未提供 GitHub 仓库或独立项目页链接，仅声明研究数据可向通讯作者合理申请；`candidate_repos` 为空）
- **核对范围：** 通读全文 1–5 节（Introduction、Related Work、Methods 3.1–3.5 及 Algorithm 1、Results and Discussion 4.1–4.7、Conclusions 5.1–5.3、Data Availability Statement 及 References 1–39），完整核对 Table 1–Table 8 与 Figure 1–Figure 16 的文字描述。
- **通读：** 已完成

## 原文核对笔记

- **核心问题：** 预训练视觉-语言模型（VLM）在直接用于长程（long-horizon）物理操作任务规划时，面临两大障碍：（1）缺乏领域任务流程先验，导致动作因果顺序等逻辑推理偏差；（2）对非结构化 RGB 图像中的同名多实例消歧与空间拓扑关系感知不足，导致语义合理但几何不可行的规划输出，且缺乏结合标准操作流程的闭环故障归因与重规划能力。
- **方法架构（ATP-VPDSG）：**
  1. **视频先验提取（Video Prior Extraction）：** 使用基于结构相似性 $D(t)=1-\mathrm{SSIM}(I_t, I_{t-1})$ 的自适应阈值与局部峰值保留采样将演示视频压缩为关键状态帧序列；定义 4 大类共 10 个基础技能原语（Spatial Motion: `Move`、`Lift`；Grasping and Placing: `Grasp`、`Place`；Assembly and Adjustment: `Align`、`Insert`；Interaction: `Push`、`Pull`、`Rotate`、`Press`）；通过角色定义、思维链（CoT）与少样本示例引导 VLM（Qwen3-VL）推断动作意图并输出 JSON 格式标准化先验知识库 $K_{\mathrm{prior}}$。
  2. **基于视觉提示的动态场景图构建（Dynamic Scene Graph Construction）：** 使用 VLM 提示 Grounding DINO v2 进行开放词汇物体检测，结合 Set-of-Mark（SoM）风格唯一数字 ID 与语义标签叠加形成标记图像 $I_{\mathrm{mark}}$，由 VLM 解析相对空间关系（left/right、front/behind、inside/on top of 等）及全局空间排序，构建 JSON 形式动态场景图 $G_t$。
  3. **双轨闭环自适应任务规划（Adaptive Closed-Loop Task Planning）：** 检索匹配先验 $K_{\mathrm{prior}}$ 并借助 $G_t$ 将泛化类别名词锚定到实例节点 ID，生成初始子任务序列 $A_t$（`temperature=0`、`max_length=4096`）。每步执行前后由 VLM 比较 $I_{\mathrm{pre}}, I_{\mathrm{post}}$ 输出二值验证信号 $\beta \in \{\mathrm{True}, \mathrm{False}\}$：若 $\beta=\mathrm{True}$，采用无感知规则化局部更新 $G_{t+1} \leftarrow \mathrm{Update}_{\mathrm{logic}}(G_t, a_t)$；若 $\beta=\mathrm{False}$，触发视觉-语义故障诊断生成错误反馈 $F_{\mathrm{err}}$，调用底层感知全量重建 $G_{t+1} \leftarrow \mathrm{Perception}(I_{\mathrm{post}})$ 并重构上下文执行重规划。
- **关键实验数据：**
  - **LIBERO-10 规划与先验提取（Table 2，每任务 50 次）：** 视频平均压缩率 70.1%（65%–74%，正文称 60–75%），平均规划准确率 91.2%（Task 8 最高 96.0%，Task 9 最低 85.3%）。
  - **故障诊断分类准确率（Table 3）：** 总体 87.3%（碰撞 92.1%、位置混淆 90.2%、抓取滑脱 88.6%、遮挡干扰 79.5%）。
  - **LIBERO-10 对比实验（Table 4，每任务 30 次、最大 600 步，底层原子技能由基准 API 与特权位姿接口实现）：** ATP-VPDSG 平均成功率 74.67%，高于 Prompt2Act（58.00%，+16.67 个百分点）、VILA（48.33%）与 ReplanVLM（46.67%）；在 10 个任务中 9 个领先或持平，仅 Task 8（80.00%）低于 Prompt2Act（86.67%）。
  - **剩余失败根因（Table 5，占总试验 25.33%）：** 底层执行误差占失败样本 45.2%（总试验 11.4%）、规划逻辑错误占 30.1%（7.6%）、场景图感知误判占 24.7%（6.3%）。
  - **消融实验（Table 6）：** 去掉视频先验降至 65.33%（-9.34 个百分点；注：正文 4.6 节文字误记为 58.33%），去掉视觉标记与场景图降至 64.33%（-10.34 个百分点）。
  - **推理延迟（Table 7，单张 RTX 5090 32 GB，仅计 VLM 推理，不含 Grounding DINO 与图像预处理）：** 离线视频先验提取 8.2–12.5 s/任务类型；在线初始建图 1.2–1.8 s；初始全序列规划 2.5–3.2 s；单步视觉验证 0.6–0.9 s；故障诊断与重规划 2.1–2.8 s。
  - **真机工业滑块-导轨装配（Table 8，6-DOF 机械臂 + 平行夹爪 + 固定外置 RGB 相机，200 mm × 35 mm 导轨，2 mm 配合公差，无任务微调，各 50 次试验）：** 30 条遥操作演示视频达 64.0%（32/50，平均成功耗时 21.32 s）；50 条演示视频达 82.0%（41/50，平均成功耗时 18.32 s）。
- **对 wiki 的映射：** [ATP-VPDSG 论文实体页](../../wiki/entities/paper-agentic-survey-doi-07c1d81a50e5849a.md)
