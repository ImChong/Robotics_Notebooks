# AesFormer_ICML2026（官方源码）

> GitHub 代码来源归档；核对日期：2026-11-15。

- **标题：** AesFormer: Transform Everyday Photos into Beautiful Memories
- **类型：** repo / image editing / two-stage aesthetic planning
- **作者 / 机构：** Tianxiang Du、Hulingxiao He、Yuxin Peng；北京大学王选计算机研究所
- **源码：** <https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026>
- **论文：** <https://arxiv.org/abs/2605.22126>；[论文归档](../papers/aesformer_arxiv_2605_22126.md)
- **模型：** 仓库 README 公告 [Hugging Face 权重与数据资源](https://huggingface.co/popo28/AesFormer)；具体版本以作者仓库链接为准
- **开源状态：** **公开了训练、两阶段推理及评测脚本**；AesRecon 数据文件需单独签协议申请，不是开放直下；外部依赖仓库和基础模型须另行安装。
- **沉淀到 wiki：** [AesFormer](../../wiki/entities/paper-aesformer.md)

## 官方入口与可核实的数据流

| 环节 | 官方 README 入口 | 实际作用 |
|---|---|---|
| Stage 1(a) | `train/Stage1_a/train_Stage1_a.yaml` → `llamafactory-cli train` | Qwen3-VL-8B 冷启动 SFT |
| Stage 1(b) | `train/Stage1_b/GRPO-A_reward_server.py`、`train_Stage1_b.sh` | EasyR1 + Qwen2.5-VL-32B 奖励服务；GRPO-A |
| Stage 2 | `train/Stage2/bash/1_cache_latents.sh`、`2_cache_textencoder.sh`、`3_train_lora.sh` | musubi-tuner 调用 `qwen_image_train_network.py` 微调编辑器 LoRA |
| 规划推理 | `test/Stage1/Aesthetic_Planning.py` | 读 `jsons/test/test.json`、poor 图；加载 Stage1 模型，以 Transformers `generate` 生成 `Editing_Actions.json` |
| 图像编辑 | `test/Stage2/Aesthetic_Editing.py` | 逐张读 `Editing_Actions.json` 与 poor 图，subprocess 调用 `src/musubi_tuner/qwen_image_generate_image.py`（带 `--control_image_path`、`--prompt`、`--lora_weight`）；保存 `*_APR.png` |
| 测评 | `test/Stage2/evaluate/{ArtiMuse,LAION-V2,Q-align,final_score_summary}.py` | 汇总审美指标，需各自配置评测环境 |

## 实际脚本核查

- `Aesthetic_Planning.py` 用 `AutoModelForImageTextToText` 与 `AutoProcessor`，生成上限 `max_new_tokens=500`；指定模型根目录是 `/path/to/your/Stage1_b_model`，使用者必须配置。
- `Aesthetic_Editing.py` 调用 `qwen_image_generate_image.py` 的默认参数包括 `--infer_steps 40`、`--guidance_scale 4.0`、分辨率 `1440 1088` 与 LoRA 参数；保存 `test/Stage2/reconstructed_images`。
- `train/Stage2/bash/3_train_lora.sh` 通过 `accelerate launch` 运行 `qwen_image_train_network.py`，使用 `networks.lora_qwen_image`、`network_dim 16`、`learning_rate 1e-5`、`max_train_epochs 2` 等脚本默认值；这不等于已在本环境成功复训。

## 获取门槛与风险

README 要求先签署 Release Agreement 并邮件联系作者才能取得 AesRecon；外部基础模型（Qwen3-VL-8B、Qwen2.5-VL-32B、Qwen-Image-Edit-2511）与工具链 LLaMAFactory、EasyR1、musubi-tuner 要自行配置。仓库入口使用示例中的占位绝对路径，部署时需校正；这里仅核对源码，不声称本地推理/训练或模型权重校验已完成。

## 参考链接

- [官方 GitHub README](https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026)
- [Aesthetic_Planning.py](https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026/blob/main/test/Stage1/Aesthetic_Planning.py)
- [Aesthetic_Editing.py](https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026/blob/main/test/Stage2/Aesthetic_Editing.py)
- [Stage 2 LoRA 脚本](https://github.com/PKU-ICST-MIPL/AesFormer_ICML2026/blob/main/train/Stage2/bash/3_train_lora.sh)
