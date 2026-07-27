# BRZ911/Thinking-in-Video

> 来源归档

- **标题：** Thinking in Video / CGDJ（官方实现）
- **类型：** repo
- **组织 / 作者：** BRZ911（Yongheng Zhang 等）
- **代码：** <https://github.com/BRZ911/Thinking-in-Video>
- **数据集：** <https://huggingface.co/datasets/BRZ911/Thinking-in-Video-Data>
- **论文：** <https://arxiv.org/abs/2607.17523>
- **入库日期：** 2026-07-27
- **一句话说明：** **Causal-Generative Dual-Judge（CGDJ）** 官方流水线：`Perception/`（扁平时空 VQA + Whisper + Gemini 判定）与 `Prediction/`（因果后果视频 + Gemini 质量分）；数据在 Hugging Face 公开。

## 开源核查（2026-07-27）

- GitHub 仓存在 `Perception/`、`Prediction/`、`figures/`、根 README；HF 数据集 **public / non-gated**。
- API 元数据 `license: null`（未见根级 LICENSE 文件）；使用时以作者 README / 引用要求为准。
- **已开源** 评测脚本与数据；生成/评判依赖外部 API（Veo 3.1 示例、Gemini-3-Pro、Whisper-large-v2）。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `Perception/perception_data.jsonl` | 显式因果感知输入表 |
| `Perception/01_generate_video_veo3.py` | Stage 1：图→视频（示例 Veo 3.1） |
| `Perception/02_transcribe_audio_whisper.py` | Stage 2：Whisper 转写 |
| `Perception/03_judge_answer_gemini3.py` | Stage 3：Gemini 判定 Correct/Incorrect |
| `Prediction/prediction_data.jsonl` | 隐式生成预测输入表 |
| `Prediction/01_generate_video_veo3.py` | Stage 1：预测后半段 |
| `Prediction/02_judge_video_gemini3.py` | Stage 2：语义对齐 / 参考一致 / 物理有效性打分 |
| HF `BRZ911/Thinking-in-Video-Data` | 配套评测数据 |

## 最短复现路径

```bash
conda create -n thinkvideo python=3.11 -y && conda activate thinkvideo
pip install -r requirements.txt   # 以仓内文件为准
export GOOGLE_API_KEY=...
cd Perception && python 01_generate_video_veo3.py && \
  python 02_transcribe_audio_whisper.py && python 03_judge_answer_gemini3.py
cd ../Prediction && python 01_generate_video_veo3.py && python 02_judge_video_gemini3.py
```

换模型时：按 jsonl 中图像跑自有生成器 → 指向 `outputs/<model>/` → 复用后续 judge 阶段，比较 `judge` 与 `score` 得 Perception-Prediction Gap。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Thinking in Video](../../wiki/entities/paper-thinking-in-video.md) | 实体归纳：CGDJ、Flatten Temporal Video、Gap |
| [物理保真度输出轴](../../wiki/overview/world-model-physics-fidelity-outputs.md) | 「画面连续 ≠ 动力学 / 因果」诊断坐标 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 视频生成器作世界模拟器的评测压力测试 |
| [Imagined Rollouts…](../../wiki/entities/paper-imagined-rollouts-kinematic-not-dynamic.md) | 另一条「想象≠动力学」诊断线（latent MBRL 侧） |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/thinking_in_video_arxiv_2607_17523.md`](../papers/thinking_in_video_arxiv_2607_17523.md)
- 沉淀 **[`wiki/entities/paper-thinking-in-video.md`](../../wiki/entities/paper-thinking-in-video.md)**
