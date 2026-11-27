# A Conversational Framework for Human-Robot Collaborative Manipulation with Distributed Generative AI models

- **作者：** Arash Ghasemzadeh Kakroudi, Roel Pieters
- **机构：** Automation Technology and Mechanical Engineering, Tampere University (33720, Tampere, Finland)
- **版本 / 载体：** arXiv:2606.06061v1 [cs.RO]（2026-06-04 提交，CC BY 4.0；已接收至 RO-MAN 2026，终版标题为 *A Distributed Conversational Framework for Human-Robot Collaborative Manipulation Using Local LLMs and VLMs*）
- **原文：** <https://arxiv.org/abs/2606.06061> ；HTML 全文：<https://arxiv.org/html/2606.06061>
- **项目页 / 代码：** <https://github.com/cogrob-tuni/franka-llm> （**已开源**：官方仓库提供 ROS 2 功能包、Web 仪表盘、`config.yaml`、启停脚本与 180 次实验结果 CSV）
- **核对范围：** 完整通读 arXiv HTML 全文（Abstract、Section I 引言与贡献、Section II 相关工作、Section III-A~E 架构/部署/NLP/视觉定位公式 (1)-(2)/通信、Section IV-A~B 平台与评测协议及 Table I、Section V-A~C 实验分析/局限/未来工作、Section VI 结论与 References [1]-[28]）；并对照 GitHub 官方仓库主页与 README 核查开源模块、配置文件与入口脚本。
- **通读：** 已完成

## 原文核对笔记

1. **核心问题与设计哲学：** 面向协作机械臂（Franka Emika FR3）的自然语言操作在真实工作台面临指令欠规范、多物体遮挡与模型幻觉风险。作者未采用端到端黑盒 VLA 策略，而是用 ROS 2 多智能体架构将**语言理解（LLM）**、**开放词汇视觉定位（VLM）**、**中央协调（Coordinator）**与**运动规划执行（MoveIt2）**解耦为独立节点，并在动作执行前通过 Web 仪表盘强制执行**人工确认闸门（explicit operator confirmation）**。
2. **本地分布式部署：** 通过 Ollama 本地推理（无云端数据外泄）。默认硬件配置为：NVIDIA Jetson AGX Orin 64 GB 运行 `ministral-3:8b`（LLM 规划），RTX 6000 工作站运行 `Qwen2.5-VL 32B`（VLM 视觉定位），也可通过修改 `config.yaml` 切换为单机或多机分布式部署。
3. **NLP 结构化解析（Section III-C）：** LLM 将自由自然语言映射为结构化动作请求（`pick`、`place`、`handover`、`go_home`）及对象/空间关系参数。问候语直接回复；涉及物体的指令路由至 VLM；`place` 的方位词（`left`/`right`/`top`/`bottom`）由协调节点叠加可配置的 `8 cm` 偏移量。
4. **零样本 VLM 定位与 2D→3D 变换（Section III-D）：** LLM 与 VLM 均在零样本设置下运行，未做任务特定微调。VLM 从腕部 Intel RealSense D435i RGB 图像返回中心像素 `(u_c, v_c)`（正则级联解析 `FOUND at pixel (u, v)` 及 `[0, 1]` 归一化坐标）。协调节点以 `(u_c, v_c)` 为中心在对齐深度图采样 `5×5` 邻域并取非零中值作为深度 `z`，结合 `sensor_msgs/CameraInfo` 内参 `(f_x, f_y, p_x, p_y)` 按针孔模型反投得到相机坐标 `p_c`（公式 1），再经离线 ArUco 标定 `(R_{a←c}, t_{a←c})`、轴向映射修正 `Φ` 与常量偏置 `d` 转换至 FR3 基坐标系 `fr3_link0`（公式 2）。
5. **基准评测与消融（Section IV & Table I，共 180 次实验）：** 3 类任务（Pick / Place / Handover）× 3 类场景歧义（Single / Multiple / Overlapped 彩色骰子）× 5 次重复 = 每配置 45 次、4 组配置共 180 次：
   - `ministral-3:8b` (Jetson) + `Qwen2.5-VL 32B` (RTX 6000)：Pick `5/5/5`、Place `5/5/5`、Handover `5/5/5`（45/45）；LLM 延迟 `5.47 s`，VLM 延迟 `9.57 s`（端到端 `15.04 s`）；文件体积 `6.0 / 21.0 GB`，运行时内存 `43.00 / 127.69 GB`。
   - `LLaMA 3.1 8B` + `Qwen2.5-VL 32B`（同机竞争）：Pick `5/5/4`、Place `5/5/5`、Handover `4/4/4`；LLM `5.97 s`，VLM 激增至 `42.83 s`；运行时内存 `28.58 / 127.69 GB`。
   - `Qwen2.5 7B` + `Llama 3.2-Vision 90B`：Pick `1/0/3`、Place `2/1/2`、Handover `3/2/0`；LLM `5.95 s`，VLM `40.69 s`；文件 `4.7 / 54.0 GB`，运行时内存 `7.65 / 116.98 GB`。
   - `DeepSeek-R1 7B` + `LLaVA 7B`：全任务 `0/0/0`，多数请求未能到达执行阶段。
6. **原文自述局限（Section V-B）：** 不支持复杂条件组合推理（如比较骰子点数大小）、非正式口语或多动作复合指令；系统无跨轮对话记忆（每条指令独立处理）；依赖通用预训练权重，专业工业/实验室器具需后续领域微调与主动澄清对话。

对 wiki 的映射：[paper-agentic-2606-06061](../../wiki/entities/paper-agentic-2606-06061.md)
