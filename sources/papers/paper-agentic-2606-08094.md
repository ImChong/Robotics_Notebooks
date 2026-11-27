# vla.cpp: A Unified Inference Runtime for Vision-Language-Action Models

- **作者：** Khanh D. Nguyen, Hung T. Ho, Chinh T. Nguyen, Thanh Q. Duong, Linh D. Le, Duy M. H. Nguyen, Vien A. Ngo, An T. Le
- **机构：** VinRobotics; Center for AI Research, VinUniversity; TU Darmstadt; Max Planck Research School for Intelligent Systems; University of Stuttgart; DFKI
- **版本 / 载体：** arXiv:2606.08094v2 [cs.RO]（2026-09-21；v1 提交于 2026-06-06，8 页，5 张图）
- **原文：** https://arxiv.org/html/2606.08094 （arXiv 摘要页：https://arxiv.org/abs/2606.08094 ）
- **项目页 / 代码：** https://github.com/VinRobotics/vla.cpp （Apache-2.0，已开源；含 C++ 运行时构建、GGUF 转换/量化文档、ZeroMQ+Protobuf 服务端、Python 绑定与 LIBERO/SimplerEnv/RoboCasa/ALOHA 评测客户端；另有配套仓库 `ggml-compiler` 与 LeRobot fork）
- **核对范围：** 通读 arXiv HTML v2 全文（Abstract、§I–§V、Eq. (1)–(6)、Table I–VIII、Fig. 1–5 及 References）与官方 GitHub 仓库主页/README（支持矩阵、构建命令、`vla-cli` / `vla-server` 入口、配套仓库及文档索引）。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与定位：** VLA 策略部署到机器人机载处理器时，往往绑定于模型各自的 PyTorch 训练栈；`vla.cpp` 基于 `llama.cpp` / `ggml` 构建统一 C++ 推理运行时，在推理阶段移除 PyTorch 依赖，以 GGUF 统一封装权重、元信息与归一化统计量，并支持 CPU、CUDA、Metal、SYCL、OpenVINO（GitHub 仓库支持矩阵亦覆盖 ROCm 与 Hexagon）。
- **运行时架构（§III）：**
  - 三层分离：模型适配器（model adapters，保留模型特定的注意力掩码、位置编码、状态 token 放置与动作头）、张量执行后端（ggml backends）、机器人客户端（负责分词、感知采集、动作映射与回放）。
  - 推理范式统一：迭代策略（flow-matching / 扩散类，如 SmolVLA、$\pi_0$、$\pi_{0.5}$、GR00T-N1.5/1.6/1.7、Evo-1、VLA-JEPA）在 $T$ 步求解器中缓存并复用观测条件 $C = F_\theta(o)$，耗时近似 $L_{\mathrm{model}}(T) \approx P + TE$；回归策略（BitVLA、OpenVLA-OFT、VLA-Adapter）直接输出 $A = R_\theta(o)$。
  - 服务端通信：`vla-server` 常驻权重，通过 ZeroMQ 与 Protocol Buffers 响应同步请求；客户端可用 Python 编写，模型执行在 C++ 侧。
  - BitVLA 三值张量核（§III-D）：权重 $q_{nj}\in\{-1,0,1\}$ 按每字节 4 个 2-bit 编码打包；激活按行量化至 INT8，使用 IMMA 张量核指令替代 `dp4a` CUDA 核，INT32 累加后缩放到 BF16。
  - 数值校验（§III-E）：`prediction-check` 固定权重、图像、token、状态、噪声、掩码、时间表与变换，校验 $e_\infty$ 与相对 RMS 误差 $e_{\mathrm{rel}}$。
- **关键实验数据（§IV）：**
  - **LIBERO-Object（Table I，100 episodes/模型）：** 11 个模型中 10 个与参考成功率差在 3.2 个百分点内（BitVLA 与 VLA-JEPA 达 100%，GR00T-N1.7 达 99%）；GR00T-N1.6 为 88% vs PyTorch 复现参考 99%（原因未解决）。
  - **四套件扩展评测（Table II，200 episodes/套件）：** BitVLA（Spatial 94.5%、Object 100%、Goal 92.5%、Long 83.5%）与 GR00T-N1.7（96.0%、98.0%、97.5%、89.0%）；SimplerEnv WidowX 上 GR00T-N1.6 在 put carrot on plate 达 80%、put spoon on towel 达 70%。
  - **边缘部署与三值加速（Table III & §IV-D）：** 8 GB Jetson Orin Nano 上 7 个被测模型有 5 个可加载（GR00T-N1.6/N1.7 OOM），BitVLA 完成 200/200 LIBERO-Object（2199 MiB 峰值 RSS，6504 MiB 系统峰值内存）。三值打包将文件从 5.6 GiB 降至 1.34 GiB、加载瞬态主机内存从 11.1 GiB 降至 1.15 GiB；IMMA 相对 `dp4a` 在 RTX 3060（172.8 → 37.85 ms/action，$4.57\times$）和 AGX Orin（406.6 → 101.11 ms/action，$4.02\times$）实现 4.0–4.6× 加速且保持 100% 成功率。
  - **服务端延迟对比（Table IV，RTX 3090）：** 相对 eager PyTorch 加速 $1.02$–$6.21\times$；相对 eager/`torch.compile` 较快者，4 个模型持平，SmolVLA $1.18\times$、GR00T-N1.5 $1.26\times$、BitVLA $1.84\times$。在 RTX PRO Blackwell 1000 上，TensorRT 运行 GR00T-N1.7（70.5 ms）比 `vla.cpp`（130.2 ms）快 $1.85\times$。
  - **GGUF 量化与求解步数（Table V & VI）：** GR00T-N1.7 在 Q4_0+V 下文件由 6292.4 MB 缩至 4556.0 MB、加速 $1.144\times$、成功率 195/200（BF16 为 196/200）。求解步数扫描呈线性 $13.9 + 5.8T$ ms；Roofline 分析显示前缀计算属计算受限（256–530 FLOPs/byte），动作专家属带宽受限（约 50 FLOPs/byte）。
  - **SmolVLA 位置索引精度案例（§IV-F）：** 图像块坐标跨索引边界导致视觉位置嵌入 $e_{\mathrm{rel}}$ 达 0.32、去归一化动作最大差约 1.97（足以翻转夹爪指令），对齐半精度参考后 LIBERO-Object task 0 成功率从 2/10 升至 9/10。
  - **真机与执行视界（§IV-G）：** UR10e（SmolVLA，$S=25$）跨 4 后端往返延迟跨度 33×（111.4–3652.9 ms），成功率 12/20–15/20；ALOHA（GR00T-N1.6 on AGX Orin）两任务 18/20 与 17/20；Orin Nano 上 SmolVLA（$H=50$）扫描 $S$ 显示 $S=16$ 达 98%（49/50），而全块开环 $S=50$ 跌至 38%（19/50）。

- 对 wiki 的映射：[vla.cpp：面向视觉-语言-动作模型的统一推理运行时](../../wiki/entities/paper-agentic-2606-08094.md)
