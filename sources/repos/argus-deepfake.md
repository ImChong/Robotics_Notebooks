# XavierJiezou/ARGUS

> 来源归档

- **标题：** ARGUS — Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection
- **类型：** repo
- **组织 / 作者：** XavierJiezou（BJTU / Tsinghua / Ant Group 合作线）
- **代码：** <https://github.com/XavierJiezou/ARGUS>
- **项目页：** <https://xavierjiezou.github.io/ARGUS/>
- **论文：** <https://arxiv.org/abs/2608.06865>
- **数据集：** <https://huggingface.co/datasets/XavierJiezou/argus-datasets>
- **权重：** <https://huggingface.co/XavierJiezou/argus-models>
- **Demo Space：** <https://huggingface.co/spaces/XavierJiezou/ARGUS>
- **入库日期：** 2026-09-07
- **一句话说明：** ARGUS 官方仓：FaceVid-Forensics-100K 使用说明、四观测者 + Judge 的 SFT/GRPO 训练、LoRA 推理与 OOD 批量评测。

## 开源核查（2026-09-07）

- README 含完整安装（`create_env.sh`）、数据下载（`hf download XavierJiezou/argus-datasets`）、帧抽取、基座模型下载、训练与推理命令。
- HF Models 提供 `weights/qwen2_5_vl_7b/main/` 下 texture/lighting/motion/physics 共享 LoRA 与 Judge（`grpo_video` 为主结果）。
- **已开源** 可运行：单视频 `src.argus_infer argus`、split JSON 批量评测、training-free 设置（README §5 后半）；训练需多卡与基座 checkpoint。

## 入口速查（对齐仓库 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `create_env.sh` | Conda `python=3.12` 依赖安装 |
| `hf download XavierJiezou/argus-datasets` | 拉取 FaceVid-Forensics-100K（≈20.9 GB） |
| `python -m src.extract_video_frames ...` | 视频 → PNG 帧树（训练/推理读帧） |
| `scripts/train_observers.sh` | 四 Observation Agent SFT |
| `python -m src.train sft --role judge` | Judge SFT（text / `--with-video`） |
| `python -m src.train grpo ...` | Judge GRPO 微调 |
| `hf download XavierJiezou/argus-models` | 预训练 LoRA 到 `weights/` |
| `python -m src.argus_infer argus --video ...` | 单视频终端判决 |
| `python -m src.argus_infer argus --input splits/ood.json ...` | OOD 批量 JSON 输出 |

## 数据目录约定

```text
data/FaceVid-Forensics-100K/
├── splits/{train,test,ood}.json
├── videos/
├── frames/          # 本地解码生成
├── observations/{raw,aggregated}/
└── explanation/{raw,aggregated}/
```

## 权重目录约定

```text
weights/qwen2_5_vl_7b/main/
├── shared/lora/{texture,lighting,motion,physics}/
├── sft_text/lora/judge/
├── sft_video/lora/judge/
├── grpo_text/lora/judge/
└── grpo_video/lora/judge/   # 主结果 Judge
```

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [ARGUS 深度伪造鉴伪](../../wiki/entities/paper-argus-deepfake-forensics.md) | 实体归纳：数据集、多 Agent 管线、OOD 表 |
| [SIDA](../../wiki/entities/sida.md) | 同属媒体鉴伪 MLLM；SIDA 图像 DET/SEG，ARGUS 视频多 Agent |
| [Daily-Omni](../../wiki/entities/paper-daily-omni.md) | 同属 MLLM 评测生态；Daily-Omni 测 AV 时序对齐，ARGUS 测伪造取证 |
| [机器人 ARGUS](../../wiki/entities/paper-argus-dynamic-symmetry.md) | **不同工作** — Sci. Robotics 对称性机器人，勿混仓库名 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/argus_arxiv_2608_06865.md`](../papers/argus_arxiv_2608_06865.md)
- 项目页：[`sources/sites/argus-deepfake-github-io.md`](../sites/argus-deepfake-github-io.md)
- 沉淀 **[`wiki/entities/paper-argus-deepfake-forensics.md`](../../wiki/entities/paper-argus-deepfake-forensics.md)**
