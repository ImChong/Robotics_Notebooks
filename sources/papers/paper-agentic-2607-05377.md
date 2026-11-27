# Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation

- **作者：** Jiaqi Peng, Xiqian Yu, Delin Feng, Yuqiang Yang, Wenzhe Cai, Jing Xiong, Ganlin Yang, Jinliang Zheng, Jiafei Cao, Xueyuan Wei, Jiangmiao Pang, Yuan Shen, Tai Wang
- **机构：** Tsinghua University; Shanghai AI Laboratory; Peking University; USTC
- **版本 / 载体：** arXiv:2607.05377v1 [cs.RO]（2026-07-06 提交）
- **原文：** [arXiv HTML v1](https://arxiv.org/html/2607.05377) / [arXiv Abs](https://arxiv.org/abs/2607.05377) / [项目页](https://steinate.github.io/cortex.github.io/)
- **项目页 / 代码：** 部分开源（官方项目页 [steinate.github.io/cortex.github.io](https://steinate.github.io/cortex.github.io/) 已上线，论文内链仓库 `https://github.com/steinate/Cortex` 对应已核验 GitHub 仓库 [Sage-PhysicalAI/Cortex](https://github.com/Sage-PhysicalAI/Cortex)；已释放 System-2 推理/评测代码、WebSocket 部署服务、三阶段子任务标注流水线、`dump_bin_bigbin` 示例数据及 Hugging Face 上的 `Steinate/Cortex` 模型权重与 4k+ 小时子任务标注集；LIBERO/RoboTwin 闭环 System-1/2 评测脚本与 System-2 训练脚本在 README TODO 中尚未发布）
- **核对范围：** arXiv:2607.05377v1 全文（Abstract、§1–§6、Table 1–3、Fig. 1–7 文字说明、Appendix A.1–A.5.8 含式 (1)–(16) 与 Table 4–10）及官方项目页摘要/结果页面
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心问题与双向对齐接口（§1, §3.1）：** 单体 VLA 受限于马尔可夫假设，在长程操作中易混淆中间进度并陷入重复动作；传统双系统方法又存在高层语言语义规划与低层运动学执行脱节的问题。Cortex（亦称 InternVLA-M1.5）构建高-低层双向对齐的具身智能体框架，核心接口为受约束子任务 $s_t$ 与主动更新的语言记忆 $M^{(t)} = M^{(0)} \oplus \bigoplus_{i=1}^{k-1}\Phi(s_i)$：
  - **可执行性（Executability）：** 将自由语言指令规范化为 **32 个标准原子技能原语**（`Pick`, `PickAndPlace`, `Place`, `Remove`, `Press`, `Push`, `Pull`, `Navigate`, `Fold`, `Wipe`, `Close`, `Open`, `Pour`, `Cut`, `Rotate`, `Handover`, `Sweep`, `Stack`, `Unstack`, `Screw`, `Unscrew`, `Scan`, `Aim`, `Clamp`, `Rinse`, `Spread`, `Release`, `Retreat`, `AdjustPosture`, `Tie`, `Strike`, `Stir`），输出固定 JSON `{"current_skill", "current_subtask", "active_language_memory"}`（结束时 `current_skill: null`、`current_subtask: "task_completed"`）。
  - **可追踪性（Tractability）：** 在数据生成中显式注入物体颜色/材质属性、基于位姿的同质物体消歧（如 `"the right white stapler"`）、计数标注及本体可达性约束（如动态插入 `handover`）。
- **数据管线与单调动态规划切分（§3.2, Appendix A.1, A.4）：**
  - 聚合 **4,000+ 小时**开源长程视频（平均每条 > 7 个子任务，含 AgibotWorld、Galaxea、BEHAVIOR-1K、RoboCerebra 及自采遥操作数据，由 `Qwen3-VL-235B` 按模板重标注）及 **30+ 小时**程序化仿真数据（RoboTwin 2.0 与 RMBench，在专家脚本 `play_once()` 内用 `record_subtask` 上下文管理器在线记录）。
  - 针对真机轨迹开发免逐帧人工标注的单调边界推断算法：融合状态-动作与视觉特征 $x_t = [\sqrt{\lambda_s}\,\phi_s(o_t, a_t), \sqrt{\lambda_v}\,\phi_v(I_t)]$，按高斯原型马氏距离 $c_k(t)$、相对时长先验惩罚 $P_k(i,j)$ 与低运动量边界惩罚 $M(j)$ 通过动态规划回溯求解严格单调边界 $b^*_{1:K-1}$（式 (1)–(13)）。
- **事件平衡采样与状态历史文本化（§3.3, Appendix A.2, A.4）：**
  - 在真值子任务边界 $t_k$ 附近定义非对称边界转换阶段 $[t_k-\epsilon_1, t_k+\epsilon_2]$（$\epsilon_2 > \epsilon_1$，总窗长约 1 s；按节奏在 RoboCerebra 设 $\epsilon=0.5\text{ s}$、Galaxea 设 $\epsilon=1.5\text{ s}$），与任务内执行阶段和尾部结束阶段形成约 **76% 任务内 + 24% 边界转换** 的采样混合比；同时支持 42.5% 详细指令 / 20.0% 显式子任务列表 / 37.5% 粗粒度目标的多粒度指令微调。
  - 在 32× NVIDIA A800 (80GB) 上用 DeepSpeed ZeRO-3 全参数解冻微调 `Qwen3-VL-8B-Instruct`（约 14.2M 多模态样本，bfloat16，AdamW，峰值学习率 $3\times 10^{-6}$，global batch size 512，Table 4）。
  - 针对重复按按钮等视觉难辨高频任务，注入最近 $H=30$ 帧截断至 $D_{\max}=32$ 维的机器人状态历史，经 $\tanh$ 压缩并 256-bin 离散化（式 (14)–(15)）连同逐步 $L_1$ 变化摘要以纯文本送入 VLM。
- **异步闭环部署与 Harness Engineering（§3.4, Appendix A.4, A.5.3）：**
  - 真机上 System-1（$\pi_{\mathrm{mem}}^{\mathrm{sub}}$，3.62B 参数）约 **10 Hz** 运行，System-2 约 **2 Hz** 异步运行。
  - 部署 Harness 负责三模式统一提示构造、标准命令模板/关键词回退映射、结合匹配置信度/最大步进/最小驻留时间的防高频抖动保持，以及感知-执行失步卡死时的超时低幅微扰复位（Timeout-driven Kinematic Reset）。
- **关键实验数据（§4, Table 1–3, Table 5, Table 8–9）：**
  - **开放环 VLM（Table 1，Qwen-3.5-9B 裁判，满分 10）：** Step-level 平均总分 **8.318**（对比 Gemini 3.1 Pro 6.925、Qwen3-VL-8B 6.739、GPT-5 6.268）；Episode-level 自回归记忆平均总分 **7.810**（对比 GPT-5 7.231、Gemini 3.1 Pro 6.860、Qwen3-VL-8B 6.292）。
  - **Galaxea 采样消融（Table 5）：** 事件平衡采样（2.72M 样本，2.23:1）对比任务内主导基线（3.10M 样本，3.77:1），总分从 **7.58** 提升至 **8.18**。
  - **LIBERO-Long 零样本（Table 2，共享微调 $\pi_{0.5}$）：** **95.5%**（超越 $\pi_{0.5}$ 的 92.4% 达 +3.1%，亦高于 Gemini-3.1-Pro 91.0%、GPT-5.4 72.0%、Qwen3-VL-8B 68.0%）。
  - **RoboTwin 2.0（Table 8，50 clean + 500 randomized）：** 整体 **86.80%**（短程 86.00%、长程 88.00%），对比 $\pi_{0.5}$ 整体 82.74%（短程 82.60%、长程 82.95%）。
  - **RMBench（Table 9，7 任务各 100 次 rollout）：** 7 任务平均 **61.9%**（对比 Mem-0 41.7%、$\pi_{0.5}$ 12.6%），其中 Rearrange Blocks 100%、Put Back Block 100%、Swap Blocks 99%、Swap T 63%、Battery Try 37%、Observe and Pick Up 14%、Press Button 20%（其余基线均 0%）。
  - **ARX ACONE 真机 20 次试验（Table 3）：** 多阶段化学液体搅拌进度 **11.0/14**、成功率 **65%**；烧杯清洗进度 **10.5/14**、成功率 **55%**（端到端 $\pi_{0.5}$ 与 $\pi_{\mathrm{mem}}$ 在两项任务上成功率均为 **0%**，人工指挥 `Human + π_mem^sub` 分别为 75% 与 70%）。
- **局限（§6）：** 纯文本记忆丢失空间坐标与视觉细节，在跨空间移动操作中易弱化物体实例对应；标准视觉编码器对高频微状态变化不够敏感，文本化历史状态与高频动态视觉的无缝融合仍待解决。
- **对 wiki 的映射：** [Cortex: A Bidirectionally Aligned Embodied Agent Framework for Long-horizon Manipulation](../../wiki/entities/paper-agentic-2607-05377.md)
