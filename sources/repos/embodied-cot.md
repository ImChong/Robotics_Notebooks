# embodied-CoT（MichalZawalski/embodied-CoT）

> 来源归档

- **标题：** Embodied Chain-of-Thought（ECoT）
- **类型：** repo
- **来源：** UC Berkeley / Stanford / University of Warsaw
- **链接：** <https://github.com/MichalZawalski/embodied-CoT>
- **论文：** <https://arxiv.org/abs/2407.08693>
- **项目页：** <https://embodied-cot.github.io/>
- **权重集合：** <https://huggingface.co/Embodied-CoT>
- **许可：** 见仓库 LICENSE（基于 OpenVLA / Prismatic）
- **入库日期：** 2026-09-09
- **一句话说明：** ECoT 官方实现：OpenVLA 分支、Bridge 训练/评测、HF 检查点与 Colab 推理。
- **沉淀到 wiki：** [`wiki/entities/paper-ecot.md`](../../wiki/entities/paper-ecot.md)

---

## 仓库入口（README）

| 组件 | 说明 |
|------|------|
| 推理 | `transformers` 加载 `Embodied-CoT/ecot-openvla-7b-bridge`；`predict_action(..., max_new_tokens=1024)` |
| Colab | [推理 notebook](https://colab.research.google.com/drive/1CzRKin3T9dl-4HYBVtuULrIskpVNHoAH) |
| 训练 | `torchrun … vla-scripts/train.py --vla.type prism-dinosiglip-224px+mx-bridge` |
| Bridge 评测 | `experiments/bridge/eval_model_in_bridge_env.py` |
| 加速 | [tensorrt-openvla](https://github.com/rail-berkeley/tensorrt-openvla) TensorRT-LLM 编译 |
| 参考硬件 | bf16 约 16 GB VRAM；4-bit 约 5 GB |

## HF 检查点

| 模型 | 说明 |
|------|------|
| `ecot-openvla-7b-bridge` | 主实验模型（Bridge ECoT） |
| `ecot-openvla-7b-oxe` | OXE 预训练 + Bridge 推理微调 |

## 开源边界（截至 2026-09-09）

- **已开源**：训练、Bridge 真机评测、推理 notebook、推理标签解析工具。
- **权重**：HF Embodied-CoT（受 Llama-2 社区许可约束）。
- **数据**：推理标注数据集见 README / HF；原始 Bridge V2 需按 OpenVLA 数据指引获取。
