# Zero2Skill: Bootstrapping Robot Skills through Autonomous Data Collection, Training, and Deployment

- **作者：** Boyuan Wang, Zhenyuan Zhang, Zhiqin Yang, Peijun Gu, Shuya Wang, Xiaofeng Wang, Xianghui Ze, Yifan Chang, Guosheng Zhao, Jiangnan Shao, Guan Huang, Hengyu Liu, Yonggang Zhang, Wei Xue, Chunyuan Guan, Chenglin Pu, Yike Guo, Xingang Wang, Zheng Zhu
- **机构：** GigaAI、University of Chinese Academy of Sciences、Hong Kong University of Science and Technology、University of Leeds、Cornell University、Tsinghua University、Nanjing University of Science and Technology、The Chinese University of Hong Kong、FAWTD
- **版本 / 载体：** arXiv:2607.14047v3 [cs.RO]（2026-07-22；v1 提交于 2026-07-15）
- **原文：** https://arxiv.org/html/2607.14047 、 https://arxiv.org/abs/2607.14047 、 https://open-gigaai.github.io/Zero2Skill
- **项目页 / 代码：** 部分开源 — 官方仓库 [open-gigaai/Zero2Skill](https://github.com/open-gigaai/Zero2Skill)（Apache-2.0，默认分支 `main`）已发布第一方采集/抓取/质检管线、运动控制辅助脚本与 4 个 Agent Skills；未重分发第三方专有 AnyGrasp 运行时（`.so`、license、checkpoint）、SAM3 与外部 HDF5 录制脚本 `COLLECT_RECORDER_PY`。
- **核对范围：** 通读 arXiv HTML 全文（Abstract、§1–§5、References、Appendix A.1–A.4、Appendix B.1–B.4、Appendix C.1–C.3，含 Algorithm 1、Table 1–5 与 Fig. 1–9 图注）、官方项目页文本（https://open-gigaai.github.io/Zero2Skill）及官方仓库 `open-gigaai/Zero2Skill` 文件清单与 README。
- **通读：** 已完成

## 原文核对笔记

- **对 wiki 的映射：** [Zero2Skill 实体页](../../wiki/entities/paper-agentic-2607-14047.md)
- **核心问题：** 长时域真机数据采集中同一类失败模式（判定标准偏差、分割漏检、抓取深度偏差、错误选臂）会在跨回合中反复出现；若纠错仅作用于当前单回合，人类监督成本将随采集时长线性增长而非随新问题数量收敛。
- **双时间尺度架构：** 基于 OpenClaw 构建人机共生智能体系统。会话内（within-session）完全通过文本改写验证提示词与持久化纠正记忆（Corrective Memory）调整行为，不修改底层策略或感知权重；会话间（between-sessions）仅通过经纠错后准则认证的轨迹数据微调下游 VLA 策略（$\pi_{0.5}$）。
- **两阶段自主采集控制：**
  1. **任务分析与规划（§3.1）：** 输入自然语言指令与相机观测，结合工具库一次生成 4 部分规划：采集例程 $c_{\mathrm{col}}$、采集验证提示词 $p_{\mathrm{col}}$、重置例程 $c_{\mathrm{res}}$、重置验证提示词 $p_{\mathrm{res}}$，由操作员确认后执行。
  2. **验证门控采集-重置循环（Algorithm 1）：** 交替执行采集阶段与重置阶段，每阶段等待场景稳定后调用 VLM 验证器（Seed1.8）输出显式 `yes`/`no` 判定及一句解释；未通过则自动重试，单阶段重试上限默认为 $N=3$。耗尽预算后停止机械臂进入 `Alert` 状态并异步通知远程操作员。独立验证重置阶段用于尽早发现静默的环境状态漂移。
- **语言干预与 Corrective Memory（§3.2）：**
  - LLM 解析器（DeepSeek-V3.2）根据当前状态与既有记忆，将自由格式自然语言反馈区分为**持久化规则**（绑定适用条件后写入结构化 Markdown 文件，包含 `trigger`、`correction`、`scope`、`source utterance` 四字段，按验证提示词、条件化抓取策略、物体优先级、重试上限分组）与**单次微调**（仅解除当前卡死，不入库）。
- **底层抓取与标注质检平台（§3.3 & Appendix A）：**
  - 硬件参考栈：双臂 Agilex Piper + 头戴对齐 RGB-D RealSense（$640\times 480$）+ 双腕 RGB 相机；配置解耦支持替换机械臂与相机。
  - `grasp-tool` 接口以五元组 `(text_prompt, arm, x_place, y_place, δ_depth)` 调用：SAM3 生成目标掩码，AnyGrasp 在全场景点云生成 6-DoF 候选抓取，经掩码投影、NMS 排序、camera-up 规范化与高架斜向接近重排后，由 Pinocchio URDF IK 求解并以 30 Hz 发布关节指令。安全工作空间守卫限制 $x \in [0.05, 0.60]\text{ m}$、$|y| \le 0.60\text{ m}$、$|\text{offset}| \le 0.05\text{ m}$。
  - 双标签体系（`execution label` + `outcome label` ∈ {success, failure, unknown}），通信/模型超时标为 `unknown`；前置 ROS topic 预检防空录制；每回合后台计算关节加加速度 $J_{\mathrm{RMS}}$、不连续计数 $D$、平滑度 $S$、亮度 $B_f$ 与美学得分 $A_f$，评级 `good`/`marginal`/`bad`，支持用更新后提示词离线重判并固化训练清单。
- **关键实证结果（§4）：**
  - **采集效率（Table 1，50 次尝试）：** Zero2Skill 成功 50/50（100.0%），遥操作时间 0.0 min，人类总工作时间 4.8 min（人工遥操作 30.0 min，比值 16%；脚本采集 31.0 min 且仅成功 35/50），TpHM 达 10.42（遥操作 1.67，约 6 倍）；总墙钟时间为 53.0 min（遥操作 35.0 min，脚本 31.0 min）。
  - **语言纠错（Table 2, Fig. 4, Fig. 5, Table 3）：** 验证器准确率在 4 个场景均提升（3 个达到 10/10，较难的 Box fruit clearing 从 0/10 升至 4/10）；8 类物体首次尝试成功率从初始规划 12.5%（5/40）经 SAM3 重提示升至 25.0%（10/40），叠加深度偏置升至 47.5%（19/40）；单独控制的语言选臂对比将首次尝试成功率从 20.0% 提至 50.0%（carambola 由 20.0% 降至 0.0%）；在线采集 4 组中 3 组累计成功率高于干预前（Banana & Chili 组从 80.00% 降至 70.59% 后回升至 76.47%）。
  - **下游策略与数据质量（Table 4–5，各 50 条微调 $\pi_{0.5}$，20 次盲测无 Agent/Verifier）：** Zero2Skill 数据与全人工遥操作数据均取得 80.0% 策略成功率（脚本数据 55.0%）；Zero2Skill 数据具有更低 RMS jerk（$191 \pm 97$ vs. $267 \pm 66$）与更高美学得分（$61.2 \pm 1.1$ vs. $55.2 \pm 1.6$），但平滑度得分较低且方差更大（$64.2 \pm 22.7$ vs. $78.1 \pm 7.7$）。
- **原文核实局限（§4.3, §4.4, §5）：** 受限于现有抓取工具与相机精度，暂不适用于复杂灵巧操作；多轮数据飞轮目前仅验证了单轮 collect–correct–consolidate–fine-tune–deploy 闭环；模糊容器边界（Box fruit）仍易使验证器误判；持久化规则在条件不匹配时会引发局部回退。
