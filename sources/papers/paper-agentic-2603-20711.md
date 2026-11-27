# RoboECC: Multi-Factor-Aware Edge-Cloud Collaborative Deployment for VLA Models

- **作者：** Zihao Zheng, Hangyu Cao, Jiayu Chen, Sicheng Tian, Chenyue Li, Maoliang Li, Xinhao Sun, Guojie Luo, Xiang Chen
- **机构：** Peking University（School of Computer Science / School of EECS）、South China University of Technology（School of Computer Science）、Beijing Normal University（School of Artificial Intelligence）
- **版本 / 载体：** arXiv:2603.20711v2（2026-04-27，v1 提交于 2026-03-21；arXiv HTML 全文）；已录用于 IJCNN 2026
- **原文：** [arXiv HTML](https://arxiv.org/html/2603.20711) / [arXiv 摘要](https://arxiv.org/abs/2603.20711)
- **项目页 / 代码：** [GitHub: zhengzihaoPKU/RoboECC](https://github.com/zhengzihaoPKU/RoboECC)（MIT 开源；Step 2.5 核查状态：**部分开源** — 已开源规划与运行时控制层源码 `src/roboecc/`、CLI 工具 `roboecc-search` / `roboecc-train-bandwidth` 及 34 个 CPU 单元测试；未包含双端实际执行后端、A100↔Orin/Thor 实测带宽轨迹及预训练 LSTM 权重）
- **核对范围：** 已通读 arXiv HTML 全文（Abstract、§I Introduction、§II Background、§III Observation and Analysis、§IV RoboECC Framework、§V Experiments 含 Table I–IV、§VI Conclusion 与 References），并对照官方 GitHub 仓库主页与 README 说明（[repos 归档](../repos/paper-agentic-2603-20711.md)）。
- **通读：** 已完成

## 原文核对笔记

- **核心问题与动机（§I, §III）：**
  1. 未经优化的 VLA 模型在边缘端仅能达到 `1~3 Hz`（§I）/ `3~5 Hz`（§II-A2）控制频率，即便应用常规加速在消费级 GPU 上也仅 `10~15 Hz`，低于具身控制常见的 `~30 Hz` 实时阈值。
  2. **挑战一（结构异构性）：** 不同 VLA 拥有不同动作头（如 OpenVLA 为 ViT + LLM + detokenizer，CogACT 为 ViT + LLM + Diffusion/DiT）。OpenVLA 末端为同构 LLM Block，延迟随切分点近似线性变化；CogACT 在结构过渡处出现延迟跳变（例如第 13 层块与第 16 层块均满足云端负载预算，但第 13 层延迟显著更低）。
  3. **挑战二（网络带宽波动与多步敏感性）：** VLA 每次输出动作序列极短且依赖多步闭环推理，网络延迟占比有时超过 40%，无法像长文本 LLM 生成那样在单次长输出中均摊网络抖动。原文 Fig. 3 示例：传输 `[1, 17, 3072]`（102 KB）时带宽从 `10 MB/s` 降至 `1 MB/s` 会使传输延迟从 `9.9 ms` 飙升至 `99.6 ms`；切换至 `[1, 17, 768]`（25.5 KB）的邻近切分点后传输延迟降至 `24.9 ms`。
- **方法设计（§IV）：**
  1. **Model-Hardware Co-Aware Segmentation（§IV-A）：**
     - 将 VLA 抽象为 `[\mathbb{S}_{enc}, \mathbb{S}_{bac}, \mathbb{S}_{dec}]`，其中 `\mathbb{S}_{enc} \in \{\mathcal{M}_{ViT}\}`、`\mathbb{S}_{bac} \in \{\mathcal{M}_{LLM}\}`、`\mathbb{S}_{dec} \in \{\mathcal{M}_{De-tokenizer}, \mathcal{M}_{MLP}, \mathcal{M}_{LSTM}, \mathcal{M}_{Diff}, \mathcal{M}_{DiT}\}`。
     - 通过 `(L_i, H_i^L, W_i)` 映射得到单层计算量 `\mathcal{C}_{compute}^{type}`（FLOPs）与访存量 `\mathcal{C}_{datamove}^{type}`（KB），结合 GPU Roofline 模型计算延迟：`T_{GPU} = \sum_i \max(\mathcal{C}_{compute}^{L_i} / (P_i \times Parallel_i), \mathcal{C}_{datamove}^{L_i} / Bandwidth_i)`（Eq. 2）。
     - 在云端负载预算 `B^{cloud}` 约束下，通过 Algorithm 1 遍历候选切分点 `S` 寻找最小总延迟 `T_{total}^S`。
  2. **Network-Aware Deployment Adjustment（§IV-B）：**
     - **网络波动预测器：** 双端部署轻量级 LSTM（论文中约 `20.1 MB`），采样间隔满足 `t_{input} < \min(t_{cloud}, t_{edge})`（Eq. 3）。
     - **参数共享池（Parameter-Sharing Pool）：** 将包含最优切分点所在 Block 的全部层同时常驻云端与边缘端（例如一个 LLaMA Block 约 `386 MB`，相对 OpenVLA 约 `16 GB` 内存占用与 Orin `32/64 GB`、Thor `64/128 GB` 内存可忽略）。
     - **细粒度动态切换：** 依据预测带宽变化量 `\Delta NB = NB_{t+1}^{pre} - NB_t^{real}` 与双阈值 `\mathcal{T}_{high}`、`\mathcal{T}_{low}` 比较：高于 `\mathcal{T}_{high}` 时在共享池内切换至激活传输量最大的层以利用带宽，低于 `\mathcal{T}_{low}` 时切换至激活传输量最小的层以压缩传输延迟。
- **关键实验数据（§V, Table I–IV）：**
  - **硬件平台（Table I）：** 云端 A100（1410 MHz, 2496 TFLOPs 4-bit, 432 Tensor Cores, 2039 GB/s）；边缘端 Jetson Orin（MAX_N, 1300 MHz, 275 TFLOPs 4-bit, 64 Tensor Cores, 204.8 GB/s）与 Jetson Thor（MAX_N, 1386 MHz, 517.5 TFLOPs 4-bit, 96 Tensor Cores, 273 GB/s）。
  - **OpenVLA + LIBERO（Table II，每任务 10 次测试均值）：** Orin 端侧独算 `1119.4 ms`（14.1 GB）、云端独算 `151.2 ms`（14.1 GB）、固定切分 `923.3 ms`、RoboECC `354.4 ms`（云端 `136.7 ms / 12.1 GB` + 边缘 `94.5 ms / 2.0 GB`），加速约 `3.16x`；Thor 端侧独算 `628.9 ms`、固定切分 `587.2 ms`、RoboECC `300.1 ms`（云端 `137.1 ms / 12.1 GB` + 边缘 `51.3 ms / 2.0 GB`），加速约 `2.10x`。
  - **CogACT + SimplerEnv（Table III）：** Orin 端侧独算 `775.3 ms`（14.5 GB）、云端独算 `111.4 ms`、固定切分 `572.5 ms`、RoboECC `236.1 ms`（云端 `81.9 ms / 12.0 GB` + 边缘 `143.2 ms / 2.5 GB`），加速约 `3.28x`；Thor 端侧独算 `429.6 ms`、固定切分 `375.4 ms`、RoboECC `192.7 ms`（云端 `82.7 ms / 12.0 GB` + 边缘 `105.7 ms / 2.5 GB`），加速约 `2.23x`。
  - **真机实验（§V-B2）：** AgileX PIPER 机械臂（采集 1,000 条样本微调），OpenVLA 端侧独算约 `1274.4 ms`（Orin）/ `667.2 ms`（Thor），RoboECC 降至 `361.8 ms`（Orin+A100）/ `321.6 ms`（Thor+A100）。
  - **消融与开销（Table IV, §V-C）：** Orin+A100（OpenVLA）下，Edge-Only `1119.4 ms` → 仅 Co-Aware Segmentation `392.7 ms` → 再加 Network-Aware Adjustment `354.4 ms`。参数共享池开销仅占 `2.55%~2.62%`，切换决策平均耗时 `10.7 ms` 而平均延迟收益为 `32.6 ms`。
- **局限（§V-C3 & 官方仓库核查）：**
  - 论文硬件模型仅覆盖 GPU 架构，尚未建模 NPU/TPU 等异构加速器。
  - 官方仓库目前仅包含规划与运行时控制层及接口适配器，真实双端跨设备推理引擎、A100↔Orin/Thor 实测带宽日志及官方训练好的 LSTM checkpoint 尚未发布。
- **对 wiki 的映射：** [RoboECC 实体页](../../wiki/entities/paper-agentic-2603-20711.md)
