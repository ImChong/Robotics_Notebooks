# galaxea_g05_arxiv_2608_11739

> 来源归档（ingest）

- **标题：** Galaxea G0.5: One Autoregressive Stream for Robot Reasoning and Action
- **短名：** G0.5 / G05
- **类型：** paper
- **来源：** arXiv abs / PDF
- **原始链接：**
  - <https://arxiv.org/abs/2608.11739>
  - <https://arxiv.org/pdf/2608.11739>
- **项目页：** <https://opengalaxea.github.io/G05/> — [`sources/sites/opengalaxea-g05.md`](../sites/opengalaxea-g05.md)
- **代码：** <https://github.com/OpenGalaxea/GalaxeaVLA> — [`sources/repos/galaxea-vla.md`](../repos/galaxea-vla.md)
- **权重 / 数据：** <https://huggingface.co/OpenGalaxea/G05>；Open-World Dataset <https://huggingface.co/datasets/OpenGalaxea/Galaxea-Open-World-Dataset>
- **作者：** Galaxea Team
- **机构：** 星海图（Galaxea / Xinghaitu (Beijing) AI Technology）
- **版本：** arXiv:2608.11739（2026-08-12 上 arXiv；项目页介绍 2026-05-31）
- **入库日期：** 2026-08-14
- **一句话说明：** 单一 Transformer 解码器在 **同一自回归流** 里发推理 token 与动作 token（VLM-as-Actor）；跨本体 **ActionCodec** + 原生 CoT + 视觉记忆。七套评测超 \(\pi_{0.5}\) / GR00T-N1.7。官方仓 + HF 权重 **已开源**（G0.5 Community License，非商用为主）。

## 核心摘录

### 1) 问题
- 主流 VLA 把预训练 VLM 当 **条件编码器**，动作由独立 flow-matching expert 出；CoT / in-context / prompt steering 只能穿过压缩瓶颈。
- 早期 AR（RT-2 / OpenVLA）把逐步离散动作塞进词表，高频/长 horizon/高维时 token 爆炸。
- Knowledge Insulation 等「防遗忘」补丁反而把 **AR 动作监督** 请回来保护 VLM——说明 AR 不是瓶颈。

### 2) 方法要点
1. **骨干：** 从 **Qwen3.5-2B** 初始化；条件段（多视角 RGB、embodiment id、指令、本体）+ 生成段（可选 CoT + 动作码），单一 next-token CE，只算生成段。
2. **ActionCodec：** 机器人拆成左/右控制、夹爪、下身，pad 到共享维，**RVQ** + 时间对比；统一 **27 维**（9+1+9+1+7）。只发 **激活 DoF 组** 的 8 个码 × \(R\) 残差轮，闲置臂不占 token。相对 FAST 的固定 DCT，本 codec **跨本体学出来**。
3. **原生 CoT：** 四类自描述目标 `Subtask:` / `BBox:` / `Trace:` / `ActionHint:`，8 种组合（含 no-CoT）加权采样；推理可开关，不必重训。
4. **视觉记忆：** ViT 每四层插分解时空注意（跟 \(\pi_{0.7}\) / MEM）；末层丢掉历史 token 限延迟；训练随机 drop 历史帧。预训练约 **6 帧 / 5 秒窗**。
5. **可选 FM 头：** 可挂 \(\pi_{0.5}\) 式 flow expert 作加速对照；主实验默认 **纯 AR**。
6. **预训练：** 14 本体机器人数据 + Web/具身 VQA，动作:VQA **4:1**；Gemini 3 / Doubao + SAM3 自动标 CoT；DROID **不进** 基础预训练，后训练时排除评测环境与物体实例。

### 3) 实验（论文 / 项目页报告摘要）

| 设定 | G0.5 | 关键对照 |
|------|------|----------|
| R1-Lite / R1-Pro 真机微调（6 设定均分） | **76.7%** SR / 129.2 process | \(\pi_{0.5}\) 53.3%；GR00T-N1.7 24.4% |
| BEHAVIOR-1K（单通才，4 epoch） | **0.3136** | \(\pi_{0.5}\) 4ep 0.2626；冠军 RLC 四 ckpt 0.2605；1ep 已 0.2904 |
| DROID 环境/物体零样本 | **82.5%** | \(\pi_{0.5}\) 57.5%；MolmoAct2 52.0% |
| LIBERO 四套件均 | **98.9%** | Xiaomi-Robotics-0 98.7%；EO-1 98.4% |
| RoboTwin 2.0 Clean/Rand 均 | **93.3%**（93.7 / 92.8） | LingBot-VA 92.2%；Fast-WAM 91.8% |
| SimplerEnv-Bridge | **87.3%** | Xiaomi-Robotics-0 79.2%；MemoryVLA 71.9% |
| PP Bench 零样本 / 50h | 跟随 65.6→84.4；成功 59.4→75.0 | 同 50h 相对 \(\pi_{0.5}\) 跟随 +15.6、成功 +9.4；加目标裁剪图跟随 **98.4%** |

- **CoT 探针：** 单阶段 PP Bench 几乎不动（+1.5）；五阶段零样本家务（Air Fryer / Cook Bacon）AR+CoT 从 1/5→3/5、0/5→2/5；同一 CoT 下 AR 比 FM expert 更跟指令。
- **GRPO：** 每任务 1 条演示后 AR 比 SDE 化的 FM 头收敛更快、终值更高、方差更低（原生 token logp）。

### 4) 开源核查（步骤 2.5）
- **项目页：** Paper / GitHub / Hugging Face / 视频齐全。
- **GitHub `OpenGalaxea/GalaxeaVLA`（2026-08-14）：** 可运行入口齐全——`scripts/run/finetune.sh`、`scripts/serve_policy.py`、`experiments/{r1lite,r1pro,droid,libero,robotwin,so100}`、`src/g05/`。许可证 **LICENSE-G0.5**（Community License：学术/个人/教育/评估；商用受限）。
- **HF `OpenGalaxea/G05`：** `g05-base` / `g05-droid` / `g05-libero` / `g05-robotwin20` / `g05-so101` + `action_tokenizer.pt`；全套约 55 GB。另有 Open-World Dataset（500+ h，单一本体，RLDS/LeRobot）。
- **结论：** **已开源**（推理、微调、评测、权重）。预训练全量数据不全公开；许可证非 OSI 商用宽松许可。

## 对 wiki 的映射

- 升格 [G0.5 论文实体](../../wiki/entities/paper-galaxea-g05.md)
- 更新 [VLA](../../wiki/methods/vla.md)、[VLA 开源复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md)、[π0.5](../../wiki/entities/paper-pi05-open-world-vla.md)、[InternVLA-A1.5](../../wiki/entities/paper-internvla-a15-unified-vla.md)

## 当前提炼状态

- [x] 架构 + 七套数字 + 开源入口
- [x] wiki 实体、时序图与交叉引用
- [x] `sources/sites/` + `sources/repos/`
