# ARGUS: Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection（arXiv:2608.06865）

> 来源归档（ingest）

- **标题：** Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection
- **短名：** ARGUS
- **类型：** paper / deepfake / video forensics / multi-agent / MLLM / benchmark
- **arXiv：** <https://arxiv.org/abs/2608.06865>（PDF：<https://arxiv.org/pdf/2608.06865.pdf>）
- **项目页：** <https://xavierjiezou.github.io/ARGUS/>
- **代码：** <https://github.com/XavierJiezou/ARGUS>
- **数据集：** <https://huggingface.co/datasets/XavierJiezou/argus-datasets>（FaceVid-Forensics-100K，README 约 20.9 GB）
- **模型权重：** <https://huggingface.co/XavierJiezou/argus-models>
- **在线 Demo：** <https://huggingface.co/spaces/XavierJiezou/ARGUS>
- **作者：** Xuechao Zou、Shun Zhang、Kai Li、Yi Zhou、Xinyu Sun、Yuhui Chen、Zhe Wu、Congyan Lang、Junliang Xing
- **机构：** 北京交通大学（BJTU，1）；清华大学（Tsinghua，2）；蚂蚁集团 Ant Group（3）
- **版本：** arXiv:2608.06865v1（2026-08）
- **入库日期：** 2026-09-07
- **一句话说明：** 发布 FaceVid-Forensics-100K（10 万段人脸视频 / 33 种合成法）与 ARGUS 多智能体鉴伪框架：四路观测 Agent（纹理/光照/运动/物理）独立取证，Judge Agent 汇总判决；全开源小 MLLM 组合在 OOD 集上超过闭源 GPT/Gemini 与专用检测器。

## 开源状态（核查，2026-09-07）

- **已开源：** 项目页 Footer / README 同时列出 GitHub、HF Dataset、HF Models、HF Space；仓库含 `create_env.sh`、观测者 SFT（`scripts/train_observers.sh`）、Judge SFT/GRPO（`python -m src.train`）、单视频与批量推理（`python -m src.argus_infer`）、帧抽取与数据目录规范。
- **数据：** HF `XavierJiezou/argus-datasets` 公开 splits / videos / observations / explanation；`frames/` 需本地 `python -m src.extract_video_frames` 一次性解码。
- **权重：** HF `XavierJiezou/argus-models` 提供各观测 LoRA 与 Judge（`grpo_video` 为主结果）。
- **边界：** 基座 MLLM（Qwen2.5-VL / InternVL）需自行从 ModelScope 等下载到 `checkpoints/`；训练全管线算力需求高，但推理与 OOD 评测脚本齐全。
- **混名：** 勿与机器人对称性工作 [ARGUS（Sci. Robotics）](../../wiki/entities/paper-argus-dynamic-symmetry.md) 混淆。

## 摘要级要点

- **问题：** 生成式 AI 深度伪造视频带来伦理与 AI 安全挑战；既有 deepfake 基准对新兴合成法覆盖不足、细粒度文本标注稀缺；单模型或单视角 MLLM/检测器难捕捉细微伪迹，OOD 泛化弱。
- **数据：** **FaceVid-Forensics-100K** — **100,000** 视频（**21,075** real + **78,925** fake），**33** 种合成法（换脸 / 重演 / 全脸生成，含 Seedance 2.0 等）；多 MLLM 聚合 + 冲突消解管线产出维度观测与判决一致解释。
- **方法：** **ARGUS** — 四路 **Observation Agent**（texture / lighting / motion / physics）独立报告；**Judge Agent**  reconcile 证据并输出真伪 + 解释；可选「仅文本报告」或「报告 + 视频帧」两种 Judge 输入。
- **OOD 评测（7,636 视频，20 个训练外生成器）：** 全文主结果 **Ours (w/ Video)** — Acc **69.87%**、Recall **81.82%**、F1 **53.28%**；超过 Gemini-3.5-Flash（63.34/58.75/46.22）、GPT-5-mini（59.31/38.70/39.00）、专用检测器 TFCU（64.28/33.44/45.20）等；**w/o Video** 仍达 67.41/65.00/51.01。
- **关键设计：** 观测独立防止早期解释锚定；Judge 权衡一致与冲突证据；附录定性显示「更长单链推理」不如「多视角独立 + 汇总」。

## 核心论文摘录（面向 wiki 编译）

### 1) FaceVid-Forensics-100K 与自动标注管线

- **链接：** Abstract；项目页 Dataset 区
- **摘录要点：** 大规模覆盖近期生成器；train / in-domain / OOD 协议分离；细粒度文本观测 + 与判决一致的 forensic explanation 由多模型聚合生成。
- **对 wiki 的映射：**
  - [ARGUS 深度伪造鉴伪](../../wiki/entities/paper-argus-deepfake-forensics.md)
  - [SIDA](../../wiki/entities/sida.md) — 同属媒体鉴伪，SIDA 偏图像 SEG/DET，ARGUS 偏视频多 Agent 推理

### 2) 四观测 + Judge 多智能体取证

- **链接：** 项目页 Multi-Agent Forensic Reasoning；Figure 1
- **摘录要点：** 单 MLLM 易忽略细微伪迹；专业化 Agent 系统性搜 cue；Judge 不只看最显著单一 artifact。
- **对 wiki 的映射：**
  - [ARGUS 深度伪造鉴伪](../../wiki/entities/paper-argus-deepfake-forensics.md)
  - [多模态 LLM 发展路线](../../wiki/overview/multimodal-llm-development.md)

### 3) OOD 榜单：小开源 MLLM 组合 > 闭源大模型

- **链接：** 项目页 Out-of-Domain Results 表
- **摘录要点：** 全开源 Qwen2.5-VL-7B 族 LoRA 组合登顶 reported metrics；闭源 Gemini/GPT 与 forensics-tuned MLLM 均未全面领先。
- **对 wiki 的映射：**
  - [ARGUS 深度伪造鉴伪](../../wiki/entities/paper-argus-deepfake-forensics.md)
  - [机器人视觉感知栈选型闭环](../../wiki/queries/robot-perception-stack-selection-loop.md) — 上游「传感器/视频可信度」诊断参考

## BibTeX

```bibtex
@misc{zou2026argus,
  title={Multi-Agent Forensic Reasoning for Generalizable Deepfake Video Detection},
  author={Xuechao Zou and Shun Zhang and Kai Li and Yi Zhou and Xinyu Sun and Yuhui Chen and Zhe Wu and Congyan Lang and Junliang Xing},
  year={2026},
  eprint={2608.06865},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2608.06865}
}
```

## 对 wiki 的映射

- 主实体页：[`wiki/entities/paper-argus-deepfake-forensics.md`](../../wiki/entities/paper-argus-deepfake-forensics.md)
- 项目页归档：[`sources/sites/argus-deepfake-github-io.md`](../sites/argus-deepfake-github-io.md)
- 代码归档：[`sources/repos/argus-deepfake.md`](../repos/argus-deepfake.md)
- 互链：[SIDA](../../wiki/entities/sida.md)、[多模态基础](../../wiki/concepts/multimodality-basics.md)、[Daily-Omni](../../wiki/entities/paper-daily-omni.md)
