# Agent as Policy for Robotic Manipulation

- **作者：** Mengzhao Jia, Yang Lin, Xixin Zhang, Zhihan Zhang, Meng Jiang
- **机构：** University of Notre Dame、University of California San Diego、San Diego State University
- **版本 / 载体：** arXiv:2609.12541v4 [cs.CL]（2026-09-11 提交 v1，2026-11-18 修订 v4；含正文 §1–§5 与 Appendix A–F，Table 1–8，Figure 1–11）
- **原文：** [arXiv HTML v4](https://arxiv.org/html/2609.12541) / [arXiv Abs](https://arxiv.org/abs/2609.12541)
- **项目页 / 代码：** 已开源（官方仓库 [`agent-as-policy-2026/agent-as-policy`](https://github.com/agent-as-policy-2026/agent-as-policy) 采用 Apache-2.0 协议，项目页 [`agent-as-policy-2026.github.io`](https://agent-as-policy-2026.github.io/)；配套真机与仿真轨迹数据集托管于 Hugging Face [`Agent-as-Policy/agent-as-policy`](https://huggingface.co/datasets/Agent-as-Policy/agent-as-policy) 与 [`Agent-as-Policy/agent-as-policy-robodojo`](https://huggingface.co/datasets/Agent-as-Policy/agent-as-policy-robodojo)，采用 CC BY 4.0 协议）
- **核对范围：** arXiv:2609.12541v4 全文（Abstract、§1–§5、Appendix A–F、Table 1–8、Figure 1–11 文字与数值）及官方仓库 [`agent-as-policy-2026/agent-as-policy`](https://github.com/agent-as-policy-2026/agent-as-policy) 的 README 与目录文件清单
- **通读：** 已完成
- **入库日期：** 2026-11-27

## 原文核对笔记

- **核心范式（§1, §3.1）：** 提出 **Agent as Policy (AGP)**，不预先生成固化控制程序（如 Code as Policies、GaP、ASPIRE），也不仅仅编排已训练 VLA/技能策略，而是让通用编码智能体（如 Codex、Claude Code）在模型权重 $\theta$ 冻结的条件下**直接作为运行时机器人策略**。先由准备智能体将视频、图像或语言指令与接口文档整理为任务定义（目标、约束、完成判据、参考材料、工具权限），再由执行智能体按 $u_k \sim \pi_\theta(\cdot \mid g, I, H_k, W_k)$ 自主请求观测、编写并运行本地 Python 程序做几何估计、发送运动指令并依据物理反馈修正策略。
- **硬件与人机接口（§3.2, §4.1.1, Appendix A.1）：**
  - **真机平台：** 单臂或双臂 6-DOF **I2RT YAM** 机械臂 + `linear_4310` 平行夹爪（内爪最大开度 95.5 mm，1 Mbit/s CAN 总线经 `i2rt` 驱动）；双臂通过标定刚体变换共享笛卡尔坐标系。
  - **相机配置：** 腕部 Intel RealSense D405（$640\times 360$ 对齐 RGB-D，30 fps）+ 顶视固定 Logitech BRIO（$1920\times 1080$ 畸变校正 RGB，30 fps），按需抓取。
  - **七项接口原语：** Python CLI 客户端与独立服务端通过工作区 JSON 通信，提供 `state`、`status`、`help`、`frames`、`move_ee`（基于 **Mink** 的 MuJoCo 差分逆运动学求关节目标）、`move_joints`、`gripper`（`[0, 1]` 归一化开合比例）；控制器独立于智能体推理异步执行轨迹，双臂可并发运动。
- **执行效率优化 `AGP_accelerate`（§3.4）：**
  - **带反馈动作批处理：** 将测量、臂运动与夹爪指令打包提交，遇到缺失深度或位姿误差超限即中止序列并回传最新观测，减少往返推理轮次。
  - **可复用几何估计：** 按标定深度把像素转为基座坐标并缓存命名目标，抵近后用腕部深度细化；当电机反馈检测到接触或主动触发接触时使受影响估计失效。
- **主实验结果（§4.1–§4.2, Table 1, Table 7）：** 使用 GPT-6 Astra（high effort，单次上限 120 分钟），覆盖 5 类真机任务共 8 个配置：
  - 四对精密装配（AutoMate 零件，径向间隙 $c\in[0.73, 2.83]$ mm，视频指令）：AGP 8/10（均时 37.2 min / 13.02M tokens / $16.62），`AGP_accelerate` 5/5（7.6 min / 2.47M tokens / $3.82）。
  - 积木搭建（图像指令，50 mm 立方块）：金字塔 AGP 10/10（21.6 min），`AGP_accelerate` 5/5（10.0 min）；双塔 AGP 10/10（20.6 min），`AGP_accelerate` 5/5（6.6 min）；六块高塔 AGP 9/10（28.2 min）。
  - 翻骰子（语言指令，6 颗）：AGP 10/10（37.9 min），`AGP_accelerate` 5/5（17.1 min）。
  - 定向抛投（语言指令）：AGP 9/10（11.1 min），`AGP_accelerate` 4/5（3.6 min）。
  - 双臂叠毛巾（视频指令）：顺序折叠 5/5（50.8 min / 18.12M tokens / $24.14），同步折叠 3/5（21.4 min / 5.06M tokens / $7.46）。
  - **加速收益：** 在全部 5 个测得配置上，`AGP_accelerate` 将平均耗时压低超过 50%，平均 token 与推理费用压低超过 60%。
- **基座对比、基线对比与经验迁移（§4.2.2–§4.4, Table 2–3, Table 8, Appendix C–D）：**
  - **不同智能体与推理强度（Table 2，两对装配）：** GPT-6 Astra 在 Low/Medium/High effort 均 5/5（9.9 / 9.2 / 9.2 min）；GPT-5.6 Sol 5/5（14.2 min，$3.94）；Terra 1/5；Luna 0/5；Claude Code 下 Claude Opus 5 达 5/5（22.2 min，$9.75），Claude Fable 5.1 达 3/5（27.7 min）。
  - **基线对比（Table 3，MuJoCo 仿真准备 + 真机 YAM）：** 两对装配上 GaP 仿真 8/25、真机 0/5，ASPIRE 仿真 25/25、真机 2/5，AGP 真机 5/5；四对装配上 GaP 真机 0/10，ASPIRE 0/10，AGP 8/10；金字塔搭建上 GaP 0/10，ASPIRE 1/10（计入准备共 280.9 min / $78.13），AGP 10/10（21.6 min / $11.69）。
  - **经验精炼与迁移（§4.3）：** 金字塔搭建在 5 版经验迭代后，均值总耗时降 17.8%、token 降 30.9%；将 Astra 经验只读迁移给 GPT-5.6 Terra 后成功率提升，成功回合均时降 34.3%、token 降 11.4%。
  - **运行时程序分析（§4.4, Appendix D）：** 65 次执行日志显示运行期自写脚本集中于几何拟合/三角测量（如 SVD 平面拟合 `fit_head.py` 与双视角最小二乘三角测量 `triangulate.py`）、组合控制调用与图像投影标注；四对装配 48.6 min 案例中规划推理占 68.2% 时长、机械臂动作占 23.5%。
- **开源权重蒸馏与本地部署（§4.5, Table 4–5, Appendix F）：**
  - 用 Astra 在 LIBERO、RoboTwin、VLA-Arena、RoboCasa 训练集（无测试集重叠）收集 **5,658 条**成功仿真轨迹，以 Slime 对 `Qwen3.6-35B-A3B` 做全语言层 SFT（冻结视觉编码器，最大序列 65,536，batch 64，lr $10^{-5}\to 10^{-6}$，第 2 epoch / step 175）。VLA-Arena 成功率由 0.8% 升至 **57.5%**（Astra 70.6%），RoboTwin 由 8.3% 升至 **62.3%**（Astra 82.5%）；未 SFT 的 `Qwen3.8-Flash-Next` 在 VLA-Arena 33.7%、RoboTwin 48.0%，并在真机四物装箱任务取得 **5/5**（582.2 s，Astra 为 5/5、413.4 s）。
- **对 wiki 的映射：** [Agent as Policy for Robotic Manipulation](../../wiki/entities/paper-agentic-2609-12541.md)
