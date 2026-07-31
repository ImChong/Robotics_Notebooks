# Wan-Video/Wan-Dancer

> 来源归档

- **标题：** Wan-Dancer（分钟级 Music-to-Dance 官方实现）
- **类型：** repo
- **组织 / 作者：** Wan-Video / Tongyi Lab · Alibaba（通义实验室 · 阿里巴巴）
- **代码：** <https://github.com/Wan-Video/Wan-Dancer>
- **权重：** HF <https://huggingface.co/Wan-AI/Wan-Dancer-14B>；ModelScope <https://www.modelscope.cn/models/Wan-AI/Wan-Dancer-14B>
- **Demo：** ModelScope Space <https://modelscope.ai/studios/Wan-AI/Wan-Dancer>
- **论文：** <https://arxiv.org/abs/2607.09581>
- **项目页：** <https://humanaigc.github.io/wan-dancer-project/>
- **基座：** Wan-I2V（见 [Wan2.1](https://github.com/Wan-Video/Wan2.1)）；推理栈内嵌 DiffSynth-Studio 组件
- **许可：** Apache-2.0
- **入库日期：** 2026-07-31
- **一句话说明：** 分层 **Global keyframe → Local refinement** 的分钟级 music-to-dance 推理仓：`gen_video_global.sh` / `gen_video_local.sh` 调 `gen_video/*.py`，权重为 **Wan-Dancer-14B**（`global_model.safetensors` + `local_model.safetensors`）。

## URL 澄清（ingest 必读）

用户常见误写 **`https://github.com/Wan-AI/Wan-Dancer-14B`**：**GitHub 上不存在该仓**（404）。正确拆分是：

| 资源 | 正确 URL |
|------|----------|
| 代码仓 | [`Wan-Video/Wan-Dancer`](https://github.com/Wan-Video/Wan-Dancer) |
| 权重（HF 组织 Wan-AI） | [`Wan-AI/Wan-Dancer-14B`](https://huggingface.co/Wan-AI/Wan-Dancer-14B) |
| 权重（ModelScope） | [`Wan-AI/Wan-Dancer-14B`](https://www.modelscope.cn/models/Wan-AI/Wan-Dancer-14B) |

## 开源状态（项目页 + 仓库核查，2026-07-31）

- **已开源：** 推理代码（Apache-2.0）+ HF/ModelScope **Wan-Dancer-14B** 权重（含 global/local DiT、Wan2.1 VAE、umT5、CLIP）+ ModelScope Studio demo。
- **训练代码：** 公开仓以推理脚本与 DiffSynth 封装为主；论文描述两阶段训练与 LoRA 定制，完整训练管线是否另仓发布需以官方后续公告为准。
- **环境口径（README）：** Ubuntu 22.04、**8×A800 80GB**、Python 3.10；依赖含 `flash_attn`、`xfuser`、`diffusers==0.34.0`、CUDA 12.4 轮子。

## 入口速查（对齐 README）

| 路径 / 命令 | 作用 |
|-------------|------|
| `pip install -e .` + README 钉定依赖 | 安装内嵌 `diffsynth` 包与推理依赖 |
| `huggingface-cli download Wan-AI/Wan-Dancer-14B --local-dir ./Wan-Dancer-14B` | 拉 14B 权重（含 `global_model.safetensors` / `local_model.safetensors`） |
| `./gen_video_global.sh` → `gen_video/gen_video_global.py` | **全局关键帧**阶段（稀疏结构 + 全曲节奏） |
| `./gen_video_local.sh` → `gen_video/gen_video_local.py` | **局部 refinement**（需 `--global_video_path`） |
| `prompt_path`（`*_global.txt` / `*_local.txt`） | 五类舞种：古典舞 / K-Pop / 街舞 / 踢踏 / 拉丁 |
| `image_path` + `music_path` | 参考形象首帧 + 输入音乐 |
| DiffSynth 示例 | [`modelscope/DiffSynth-Studio` `Wan-Dancer-14B-local.py`](https://github.com/modelscope/DiffSynth-Studio/blob/main/examples/wanvideo/model_inference/Wan-Dancer-14B-local.py) |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Wan-Dancer](../../wiki/entities/paper-wan-dancer.md) | 实体归纳：分层 music-to-dance、分钟级连贯 |
| [Wan](../../wiki/entities/paper-wan-video.md) | 开源视频基座；本文微调 Wan-I2V |
| [Wan-Move](../../wiki/entities/paper-wan-move.md) | 同族可控 I2V：轨迹刷 vs 音乐驱动舞蹈 |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 长时程分层生成对视频先验谱系的补充 |

## 对 wiki 的映射

- 论文摘录：[`sources/papers/wan_dancer_arxiv_2607_09581.md`](../papers/wan_dancer_arxiv_2607_09581.md)
- 项目页：[`sources/sites/wan-dancer-project.md`](../sites/wan-dancer-project.md)
- 沉淀 **[`wiki/entities/paper-wan-dancer.md`](../../wiki/entities/paper-wan-dancer.md)**
