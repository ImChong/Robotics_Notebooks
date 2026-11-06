# OpenVLA 论文通读（开源 VLA 配方与微调视角）

> 来源归档（原文通读 · 支撑 `wiki/methods/vla.md`）

- **标题：** OpenVLA: An Open-Source Vision-Language-Action Model
- **作者：** Moo Jin Kim, Karl Pertsch, Siddharth Karamcheti 等（Stanford / UC Berkeley / TRI / Google DeepMind / Physical Intelligence / MIT）
- **类型：** paper / vla
- **原文：** <https://arxiv.org/abs/2406.09246>（2026-11 通过 arXiv PDF 获取全文）
- **一句话说明：** 7B 开源 VLA：Prismatic VLM（SigLIP+DinoV2+Llama 2）在 Open X-Embodiment 970k 轨迹上微调，并系统研究 LoRA 微调与量化推理。
- **核对范围：** 2026-11 重新下载 arXiv PDF 全文（pdftotext），通读摘要、§1–§6 正文（含表 1、表 2、图 3–6 的数字）及全部附录：A 数据混合（表 3）、B.1 Bridge WidowX 17 任务描述、B.2 Google Robot 12 任务、B.3 Franka-Tabletop/DROID 任务与表 7–8、D.1–D.4 消融（表 9–11）、E LIBERO 仿真（表 12）。附录 C（数据清洗）以正文 §3.3 及 E.1 的清洗步骤交叉核对；表 4–6 的逐任务数值仅浏览总体结论，未逐格核对。
- **通读：** 已完成

## 核心摘录（均出自原文）

- **架构（§3.1、图 2）：** 视觉编码器（SigLIP + DinoV2，通道拼接，约 600M）→ 2 层 MLP 投影 → Llama 2 7B；基于 Prismatic-7B（用 LLaVA 1.5 数据混合微调过的 VLM）。
- **动作 token 化（§3.2）：** 每维独立离散为 256 bin，bin 宽按训练数据 1%–99% 分位数均分（避免离群值拉大区间）；用 Llama tokenizer 中最少使用的 256 个 token 覆盖；标准 next-token 预测，仅对动作 token 算交叉熵。
- **数据（§3.3）：** 从 Open X-Embodiment（>70 个数据集、>200 万轨迹）筛选单臂末端控制、含第三人称相机的子集，约 970k 轨迹，沿用 Octo 混合权重；DROID 以 10% 权重加入但动作 token 精度一直偏低，最后三分之一训练中移除。
- **设计经验（§3.4）：** Prismatic 骨干优于 LLaVA、IDEFICS-1；224px 与 384px 效果无差异（后者训练慢 3 倍）；与 VLM 常规做法相反，微调视觉编码器对 VLA 至关重要；需训练远多于 LLM 的 epoch（动作 token 精度超 95% 后真机仍提升，最终 27 epoch）；固定学习率 2e-5，无需 warmup。
- **训练/推理资源（§3.5）：** 64 张 A100 训练 14 天（约 21,500 A100 小时），batch 2048；bf16 推理约 15GB 显存，RTX 4090 上约 6 Hz（无编译、无推测解码）。
- **直接评测（§5.1）：** 29 个任务、多种具身；绝对成功率比 RT-2-X（55B）高 16.5%，参数少 7 倍；RT-2-X 在语义泛化上更强（其与网络数据 co-fine-tune）。
- **新任务微调（§5.2）：** 10–150 条示教全参数微调；窄任务上 Diffusion Policy 更平滑精确，多指令、多物体语言条件任务上 OpenVLA 更好；OpenVLA 是唯一在所有测试任务上成功率都不低于 50% 的方法。
- **参数高效微调（§5.3、表 1）：** LoRA(r=32) 成功率 68.2% vs 全量 69.7%，仅训练约 1.4% 参数（97.6M vs 7,188M）；只训最后一层 30.3%、冻结视觉编码器 47.0%（说明视觉特征需适配目标场景）；单张 A100 上 10–15 小时，约比全量省 8 倍算力。
- **量化（§5.4、表 2）：** int4 成功率 71.9%（bf16 71.3%），显存 7.0GB vs 16.8GB；int8 因速度下降（A5000 上 1.2 Hz，偏离 5 Hz 控制器的数据采集动力学）成功率降到 58.1%。
- **局限（§6）：** 仅支持单图观测（无多图、本体感知、观测历史）；吞吐不足以支撑 ALOHA 50 Hz 等高频控制，可考虑 action chunking、推测解码；可靠性仍常 <90%；VLM 规模、是否与网络数据 co-train、最佳视觉特征等问题未探索。
- **与动作块的关系（§5.2）：** 作者指出 Diffusion Policy 的 action chunking + 时间平滑可能帮助 OpenVLA 提升灵巧性，列为未来方向。

## 附录摘录（均出自原文附录）

- **数据混合（附录 A、表 3）：** Bridge 13.3%、Fractal 12.7%、Kuka 12.7%、DROID 10.0%（最后三分之一训练移除并重分配）、BC-Z 7.5%、FMB 7.1% 等 25 余个 OXE 子集。
- **OXE 消融（D.1、表 9）：** 仅用 Bridge 训练的 OpenVLA-Bridge 在 8 个 Bridge 任务上 45.6% vs OpenVLA 76.3%（绝对下降约 30 个百分点）。
- **双视觉编码器（D.2）：** 去掉 DinoV2 的 OpenVLA-Bridge-SigLIP 为 40.6%（比 OpenVLA-Bridge 低约 5 个百分点），作者称 DinoV2 的低层空间特征仅在部分情形帮助泛化。
- **冻结 vs 微调视觉编码器（D.3、表 10）：** 两者均有成对结果的任务上，微调 80.0% vs 冻结 46.7%；冻结时常出现不稳定行为。
- **量化再评（D.4、表 11）：** 阻塞式控制下 bf16 70.0%、int8 74.4%、int4 68.8%，误差棒重叠，证实 int8 在正文中的下降源于推理变慢而非精度损失。
- **Franka 微调（B.3、表 7）：** 单指令窄任务 Diffusion Policy 更强，多指令任务 OpenVLA/Octo 更强；Franka-DROID 上 OpenVLA 最好；10–150 条示教、评测 10–12（域内）/5–6（OOD）次。
- **LIBERO（附录 E、表 12）：** LoRA(r=32) 微调的 OpenVLA 平均 76.5% 排名 1.5，Octo 75.1%、Diffusion Policy 72.4%；差距小于真机，作者归因于预训练无仿真数据；数据处理含 256px 重生成、去除 no-op 动作（对单步表达力强的策略至关重要）、剔除失败示教。

## 对 wiki 的映射

- [`wiki/methods/vla.md`](../../wiki/methods/vla.md)
- [`wiki/entities/paper-openvla.md`](../../wiki/entities/paper-openvla.md)
