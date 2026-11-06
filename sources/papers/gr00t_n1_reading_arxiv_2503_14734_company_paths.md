# GR00T N1 论文通读（公司技术路线对照视角）

> 来源归档（原文通读 · 支撑 `wiki/comparisons/robot-foundation-model-company-paths-2026.md`）

- **标题：** GR00T N1: An Open Foundation Model for Generalist Humanoid Robots
- **作者：** NVIDIA（GEAR Lab 等）
- **类型：** paper / humanoid / VLA / foundation model
- **原文：** <https://arxiv.org/abs/2503.14734>（2026-11 通过 arXiv PDF v2 获取全文，pdftotext 提取）
- **一句话说明：** 公司路线页里「NVIDIA」一行的一手出处：双系统 VLA（Eagle-2 VLM + DiT flow-matching 动作头）、数据金字塔（网络与人视频 / 合成 / 真机），并公开 GR00T-N1-2B。
- **核对范围：** 通读摘要、§1–§6 全部正文（模型架构、训练数据生成、预训练数据集、仿真与真机评测、局限、相关工作、结论）及附录 A–F（含表 4–7 逐任务结果与数据统计）；附录 A 作者名单与参考文献仅浏览，图中数值只读正文与图注文字，未逐点读图。未阅读 NVIDIA 官方白皮书及后续 N1.5 / N1.7 报告，这些版本的结论不由本归档支撑。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **双系统结构（§1–§2.1）：** System 2 为 Eagle-2 VLM（SmolLM2 + SigLIP-2，10Hz，L40），取 LLM 第 12 层中间表示而非末层（速度与成功率均更好）；System 1 为 DiT，flow-matching 生成 H=16 的动作块，交替 cross-attention（视觉语言 token）与 self-attention（状态与带噪动作），K=4 步 Euler 去噪；两者联合端到端训练。GR00T-N1-2B 共 2.2B 参数（VLM 1.34B），采样一个 16 步动作块 63.9ms；文中称 System 1 以 120Hz 闭环出动作。
- **跨本体（§2.1）：** 每个本体一套状态编码 MLP 和动作解码 MLP，投影到共享维度；相关工作节称相比 π0 的 mixture-of-experts 桥接，这里用简单 cross-attention。
- **数据金字塔（图 1、§2.2、§3、表 7）：** 底层网络数据与人类第一视角视频（Ego4D 等 7 个数据集，约 2517 小时），中层合成（仿真 1743 小时、神经视频 827 小时），顶层真机（含自采 GR-1 遥操作 88 小时、Open X-Embodiment、AgiBot-Alpha 约 1979 小时）；总计约 8376 小时。
- **无动作数据怎么用（§2.2）：** 对人视频与神经视频，用 VQ-VAE 学潜动作（LAPA 式，视为一个独立「本体」）；神经轨迹另用逆动力学模型（IDM）打伪动作标签。
- **合成数据（§2.2、§3.2）：** 微调图像到视频模型（WAN2.1-I2V-14B + LoRA）把 88 小时遥操作扩成 827 小时反事实视频，并用多模态 LLM 过滤与重标注；DexMimicGen 在 11 小时内生成 78 万条仿真轨迹（约 6500 小时人演示等价）。
- **训练规模（§2.3）：** 预训练约 5 万 H100 GPU 小时、最多 1024 卡；后训练冻结语言部分，神经轨迹与真机轨迹按 1:1 采样；单张 A6000 可做受限微调。
- **仿真结果（表 2）：** 每任务 100 条演示，RoboCasa / DexMG / GR-1 三个基准平均成功率 GR00T-N1-2B 45.0%，Diffusion Policy 33.4%，BC-Transformer 26.4%；演示更多时差距缩小。
- **真机结果（表 3、5）：** GR-1 人形桌面任务，全量数据平均 76.8% 对 Diffusion Policy 46.4%；仅 10% 数据时 42.6% 对 10.2%，接近 DP 全量（46.4%）。个别任务（如 Machinery Packing 10% 数据）反而低于 DP。预训练模型零样本两项任务 76.6% / 73.3%（各 15 次）。
- **神经轨迹消融（图 9）：** 低数据下 co-train 神经轨迹，RoboCasa 平均 +4.2 / +8.8 / +6.8 个百分点（30 / 100 / 300 演示），真机 +5.8；文中称为「初步结果」。
- **局限（§4.6）：** 主要是短时程桌面操作，长时程 loco-manipulation 留作未来；合成数据在多样性、反事实与物理合理性上仍受限。另外后训练只含右手数据时，模型丢掉了预训练模型能做的双手递交行为（§4.5）。
- **开放（§1）：** 公开 GR00T-N1-2B 检查点、训练数据与仿真基准（GitHub / HuggingFace）。

## 对 wiki 的映射

- [`wiki/comparisons/robot-foundation-model-company-paths-2026.md`](../../wiki/comparisons/robot-foundation-model-company-paths-2026.md)
- 关联实体：[`wiki/entities/isaac-gr00t.md`](../../wiki/entities/isaac-gr00t.md)
