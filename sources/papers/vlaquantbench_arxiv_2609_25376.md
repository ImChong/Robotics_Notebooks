# VLAQuantBench 论文来源归档

> 来源归档（paper）

- **标题：** VLAQuantBench: Closed-Loop Evaluation of Post-Training Quantization for Vision-Language-Action Models
- **类型：** paper
- **作者：** Jiuyi Xu, Qing Jin, Meida Chen, Song Wang, Yang Sui, Yangming Shi
- **机构：** Colorado School of Mines, Independent Researcher, University of Central Florida, USC Institute for Creative Technologies, Microsoft AI
- **arXiv：** <https://arxiv.org/abs/2609.25376>（核对版本 `arXiv:2609.25376v1`，2026-09-29 提交）
- **PDF：** <https://arxiv.org/pdf/2609.25376>
- **代码：** <https://github.com/jiuyixu25/VLAQuantBench>（MIT License）
- **入库日期：** 2026-10-18
- **一句话说明：** 通过统一 RTN 配方、独立与联合组件作用域分解、激活校准与实际 GPU 算子评测，在 94,574 条闭环仿真回合与 1,200 条 Franka Research 3 真机回合上系统揭示 VLA 量化中的“假性容忍”、动作头非单调交互、跨模型校准分化与显存-延迟悖论。
- **沉淀到 wiki：** [`wiki/entities/paper-vlaquantbench.md`](../../wiki/entities/paper-vlaquantbench.md)
- **通读：** 已完成
- **核对范围：** 通读 arXiv `2609.25376v1` 全文与全部附录（Abstract, §§1–8, Tables 1–35, Figures 1–4, Appendices A–L），重点核查 Table 1/2（四模型 LIBERO 端到端与组件精度下限）、Table 3/29/30/32（$\pi_{0.5}$ 动作头 167 层非单调嵌套交互）、Table 4/5（OpenVLA-OFT `fc2` 单层敏感性）、Table 6/7（`alpha=0.5/0.75` 平滑+裁剪跨模型分化）、Table 8/9（CALVIN 与 SimplerEnv 跨域实验）、Table 10/21（RTX 4090 真实量化 kernel 显存与延迟）、Table 11（Franka Research 3 1,200 条真机实验）及 Appendix I/L（评测栈三类隐蔽缺陷与同观测重放诊断）。

## 开源状态

- **已开源**（2026-11-14 全文与仓库交叉核查）：官方代码仓库 <https://github.com/jiuyixu25/VLAQuantBench> 以 MIT License 开源完整量化框架、7 模型与 4 基准适配器、校验脚本及 417 cells / 105,868 episodes 全量结果日志；复现实验需自行安装各基座环境脚本（如 `mujoco==3.3.2`）并下载模型权重。

## 全文核心核查摘录

- **核心问题：** 传统仅看 LLM 骨干 PPL、静态权重误差或单组动作回归损失的离线量化评估在闭环机器人控制上经常失真——例如单帧观测上 94% 与 96% 的表面指标在多步反馈后会分化为完成任务与彻底卡死；且不同研究对量化格式、作用组件与 LM head 排除规则各异，无法横向比较。
- **主评测规模与模型矩阵：** 论文主分析涵盖 **409 次评测运行、94,574 条闭环仿真回合**（附带 1,200 条 Franka Research 3 真机回合）。主测试模型为 **`π0.5`**、**`π0`**、**`OpenVLA-OFT`**、**`X-VLA`**，辅以 **`OpenVLA`**、**`CogACT`** 与 **`InternVLA-M1`**（CALVIN / SimplerEnv，Table 8–9）。
- **四模型耐受度与端到端表现（Table 1，LIBERO 200 ep/suite，X-VLA 190 ep）：**
  - `π0.5`：Spatial 基线 `99.0%`，端到端 `W3=99.0%`、`W4=97.5%`、`W8=98.5%`、`W4A8=99.0%`、`W8A8=98.5%`，但 `W4A4=0.0%`（Object `1.0%`、Goal `1.5%`、Long `0.0%`）。
  - `OpenVLA-OFT`：四套件基线 `100.0 / 97.5 / 96.0 / 92.5%`；端到端 `W3` 在 Long 崩塌至 `8.0%`，`W4A4 / W4A8 / W8A8` 几乎全线塌陷至 `0–13%`（Goal `W8A8 = 13.0%`）。
  - `π0`：基线 `76.5 / 80.5 / 72.5 / 54.0%`，`W3` 在 Long 降至 `37.5%`，`W4A4` 全套 `0%`（除 Spatial 略低 `W8A8=64.0%`），组件加和退化明显。
  - `X-VLA`：基线 `95.8 / 98.4 / 97.4 / 92.6%`，`W4A4` 在四套件归零，但 `W4A8` 仍保持 `92.6 / 98.4 / 95.8 / 88.4%`。
- **关键机制发现：**
  1. **单层 bottleneck（Table 4–5）：** `OpenVLA-OFT` 的动作头 `fc2` 输出层（仅 28,672 参数）主导了端到端崩溃；在 442 层中仅保护这 1 层，`W3` Long 从 `8.0%` 恢复至 3 种子均值 `92.0%`（`93.5/91.5/91.0%`），`W4A8` 与 `W8A8` 四套件分别恢复至 `94.0–98.0%` 与 `93.5–99.0%`。
  2. **非单调子模块抵消（Table 3, 29–35）：** 在 `π0.5` 的 167 层动作头 `W4A4` 消融中，单独量化 `attention`（72 层）得 `98.0%`、单独 `adaRMS`（36 层）得 `93.5%`，两者联合（108 层）却暴跌至 `1.5%`（`+down_proj` 126 层仅 `7.0%`），但继续量化至全头 167 层反而回升至 `70.5%`（主要靠纳入 5 个输入/条件/输出映射层，131 层组达 `84.5%`，而非 MLP `gate/up`）。同观测重放（20 条轨迹、219 chunk）证实全头 167 层的归一化 chunk MAE（`0.390`）在 99.1% 的 chunk 上低于 126 层子集（`0.514`），证明存在局部误差反向相消。
  3. **激活校准（Smooth+Clip）跨模型分化（Table 6–7）：** 仅用 2 条不相交 episode（init state 20–21）采集统计并应用 `α=0.5/0.75` 平滑+裁剪，`π0.5` 动作头从 `70.5%` 跃升至 `98.5%`（108 层联合 `1.5% -> 97.5%`，`e2e` 在 `α=0.75` 达 `91.5%`）；但相同校准使 `π0` **全线恶化**（`e2e` `61.0% -> 10.5%`，LLM `72.0% -> 11.0%`）。
- **真实算子显存-延迟悖论（Table 10 & Table 21，RTX 4090 bs=1，OpenVLA-OFT LLM）：** 仅靠 fake-quant 会掩盖真实延迟——`BF16` 峰值显存 `15.0 GiB`、控制调用延迟 `18.8 ms`（新 chunk 推理 `139.9 ms`）；采用真实底层算子后，`AWQ` 显存降至 `6.1 GiB`、`bitsandbytes NF4` `6.3 GiB`、`LLM.int8()/SmoothQuant` `9.0 GiB`，但控制与推理延迟**全面变慢**（控制延迟升至 `20.2–48.3 ms`，新推理延迟升至 `152.6–379.5 ms`，其中 `LLM.int8()` 最慢，受 bs=1 解包与非连续小矩阵开销拖累）。
- **真机验证与评测栈 pitfalls（Table 11, Sec 6, App. H–I）：** Franka Research 3（1,200 rollouts）显示 `W8`/`W8A8` 接近 `BF16`（差距 ≤6 pp），而 `W3` 剧跌（`π0.5` Pick `64% -> 12%`），`W4A8` 相对 `W4` 仍下降 `18–40 pp`。论文还披露了三个易导致假结论的工程缺陷：`mujoco >= 3.4.0` 与 `3.3.2` 碰撞结算差异使 LIBERO-Spatial 失真 3–9 pp；动作 chunk 执行步数（50 步 vs 10 步重规划）使 `π` 家族掉点 10–25 pp；以及早期未覆盖动作头激活 hook 曾导致伪造的“W8/A8 全兼容”假象。

## 对 wiki 的映射

- 实体主页：[`paper-vlaquantbench`](../../wiki/entities/paper-vlaquantbench.md)
- 上游关联主题：[`vla`](../../wiki/methods/vla.md)、[`action-chunking`](../../wiki/methods/action-chunking.md)、[`simulation-evaluation-infrastructure`](../../wiki/concepts/simulation-evaluation-infrastructure.md)、[`manipulation`](../../wiki/tasks/manipulation.md)
