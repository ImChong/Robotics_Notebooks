# Go-with-the-Track: Video Compositing and Motion Control with Point Tracking（arXiv:2606.20891）

> 来源归档（ingest）

- **标题：** Go-with-the-Track: Video Compositing and Motion Control with Point Tracking
- **类型：** paper / video-generation / point-tracking / compositing / motion-control / diffusion / wan
- **arXiv abs：** <https://arxiv.org/abs/2606.20891>
- **项目页：** <https://eyeline-labs.github.io/Go-with-the-Track/>
- **代码：** <https://github.com/Eyeline-Labs/Go-with-the-Track>（**已开源**，Apache-2.0；见 [repos/go-with-the-track.md](../repos/go-with-the-track.md)）
- **模型：** <https://huggingface.co/Eyeline-Labs/Go-with-the-Track>
- **数据集：** <https://huggingface.co/datasets/Eyeline-Labs/Go-with-the-Track>
- **机构：** Eyeline Labs、Netflix、牛津大学（Oxford）、UCLA、Stony Brook、哥伦比亚大学（Columbia）
- **会场：** SIGGRAPH 2026
- **入库日期：** 2026-09-15
- **一句话说明：** 在 **Wan2.2** 视频扩散骨干上，用 **reference-anchored point-tracks** 统一 **运动控制** 与 **多参考图合成**：把点轨迹锚定到参考帧，建立生成帧与参考内容的显式对应，支持任意关键帧条件、网格风格化、相机重定向与动态场景合成。

## 相关资料（策展）

| 类型 | 链接 | 说明 |
|------|------|------|
| 项目页 | [eyeline-labs.github.io/Go-with-the-Track](https://eyeline-labs.github.io/Go-with-the-Track/) | 应用 demo、消融、数据集可视化 |
| 代码 | [Eyeline-Labs/Go-with-the-Track](https://github.com/Eyeline-Labs/Go-with-the-Track) | `run_inference_dataset.py`；480P/720P 配置 |
| HF 模型 | [Eyeline-Labs/Go-with-the-Track](https://huggingface.co/Eyeline-Labs/Go-with-the-Track) | Wan2.2 微调 checkpoint |
| HF 数据 | [datasets/Go-with-the-Track](https://huggingface.co/datasets/Eyeline-Labs/Go-with-the-Track) | 评测集与训练样本格式 |
| 上游 | [Wan2.2](https://github.com/Wan-Video/Wan2.2) | T2V 基础模型 |

## 开源状态（项目页核查，2026-09-15）

- **判定：已开源。** 项目页链 GitHub；README 提供 conda 环境、`hf download` 拉 checkpoint 与 eval 数据集；推理脚本 `run_inference_dataset.py` + `configs/480P.sh` / `720P.sh`。
- **许可：** Apache-2.0（README badge）。
- **部分依赖：** 基础 Wan2.2 权重需从 HF 另行下载至 `checkpoints/ckpt_wan/`。

## 摘要级要点（项目页 + README）

- **问题：** 运动控制（轨迹条件 I2V）与合成（多参考图插入）常被拆成两条线；常规 point-track 只描述生成序列内 2D 流，难在多参考、非首帧条件下精细合成。
- **回答：** **reference-anchored point-tracks**——轨迹锚定在参考图上，显式建立生成帧与参考内容的点对应；联合条件 **1–4 张参考图** + **track.npy** + 可选源视频。
- **关键设计：** spatially-aware point-track embedder、point-track adapter、relative position injection；迭代 resampling 得更均匀轨迹。
- **应用：** 多参考风格迁移、网格顶点轨迹风格化、关键点合成、静态/动态场景相机控制、首帧重建与去闪烁等。

## 核心摘录（面向 wiki 编译）

### 1) 输入格式（推理）

```
{example}/
  ref/{id}.png + {id}.npy   # 1–4 参考图及参考锚定轨迹
  track.npy                 # 生成帧点轨迹条件
  stylized_prompt.txt
  video.mp4 (optional, 49 frames)
```

### 2) 训练数据策略（项目页消融）

- 真实视频 + 离线 tracker（含噪）+ **合成/静态场景 GT 轨迹** 混合训练，显著提升运动可控性。
- 四帧均匀采样参考优于仅首帧条件。

### 3) 推理规模

- 默认 **49 帧**；**480P**（1 GPU）与 **720P**（4 GPU，自 480P ckpt 再训 4000 iter）。

## 对 wiki 的映射

- 新建 [paper-go-with-the-track.md](../../wiki/entities/paper-go-with-the-track.md)。
- 互链：[generative-world-models.md](../../wiki/methods/generative-world-models.md)、[paper-wan-video.md](../../wiki/entities/paper-wan-video.md)、[video-as-simulation.md](../../wiki/concepts/video-as-simulation.md)。

## 参考来源

- Namekata et al., *Go-with-the-Track: Video Compositing and Motion Control with Point Tracking*, SIGGRAPH 2026, [arXiv:2606.20891](https://arxiv.org/abs/2606.20891)
- [项目页](https://eyeline-labs.github.io/Go-with-the-Track/)
- [Eyeline-Labs/Go-with-the-Track](https://github.com/Eyeline-Labs/Go-with-the-Track)
