# nv-tlabs/Gamma-World

> 来源归档

- **标题：** Gamma-World: Generative Multi-Agent World Modeling Beyond Two Players（官方实现）
- **类型：** repo
- **组织：** NVIDIA Toronto AI Lab（nv-tlabs）
- **代码：** <https://github.com/nv-tlabs/Gamma-World>
- **论文：** <https://arxiv.org/abs/2605.28816>
- **项目页：** <https://research.nvidia.com/labs/sil/projects/gamma-world/>
- **Hugging Face Papers：** <https://huggingface.co/papers/2605.28816>
- **入库日期：** 2026-05-30
- **开源状态：** **已开源**（GitHub 官方仓库 `nv-tlabs/Gamma-World`，Apache-2.0，Hugging Face 模型权重 `chijw/Gamma-World`）
- **核对范围：** 已核对仓库目录树、`docs/inference.md`、`docs/training.md`、`scripts/inference.py`（全文件入口、`load_action`、`load_example_sample`、`generate_sample`）与 `scripts/train.py`、`scripts/convert_checkpoint_to_dcp.py`。
- **一句话说明：** **γ-World** 配套开源仓库，提供双向与自回归（causal few-step）多智能体视频世界模型推理与训练脚本。

## 代码入口与模块核对

| 文件路径 | 入口 / 类 / 函数 | 作用 |
|------|------|------|
| `scripts/inference.py` | `parse_args()` / `load_example_sample()` / `generate_sample()` | 加载首帧、`prompt.txt` 与每玩家 `action_{idx}.json`（23 维键盘 + 2 维相机），调用 `BidirectionalInference` 或 `I2VInference`（`MODEL_SPECS`）生成多机交互视频 |
| `scripts/train.py` | `cosmos_oss.scripts.train.main` | 调用 Cosmos 训练主循环，加载 `gamma_world/_src/gamma_world/configs/{causal_cosmos2,self_forcing}/experiment/exp_mc*.py` |
| `scripts/convert_checkpoint_to_dcp.py` | DCP checkpoint 转换脚本 | 将 Hugging Face `chijw/Gamma-World` `.safetensors` 权重转换为 PyTorch DCP 分布式训练断点 |
| `gamma_world/_src/gamma_world/inference/` | `BidirectionalInference` / `I2VInference` / `MODEL_SPECS` | 实现多智能体双向全序列生成与 causal few-step 24 FPS 长时序自回归推理 |
| `gamma_world/_src/gamma_world/datasets/` | `solaris_data.py` / `multiview.py` / `sil_dataset.py` | 多玩家共享世界（Minecraft / 多机多视角）动作解析与数据流读取 |

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Gamma-World（多智能体世界模型）](../../wiki/entities/paper-gamma-world-multi-agent.md) | 实体页：SRAE、Sparse Hub、教师–学生蒸馏与 24 FPS |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | 像素域世界模型谱系中的 **多智能体 / 置换对称** 分支 |
| [机器人世界模型训练闭环 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) | ③ 可控视频生成 · 多机共享环境案例 |

## 对 wiki 的映射

- 沉淀 **[`wiki/entities/paper-gamma-world-multi-agent.md`](../../wiki/entities/paper-gamma-world-multi-agent.md)**；原始论文摘录见 [`sources/papers/gamma_world_arxiv_2605_28816.md`](../papers/gamma_world_arxiv_2605_28816.md)。
