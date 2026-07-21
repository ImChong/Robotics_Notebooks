# RynnBrain

> 来源归档

- **标题：** RynnBrain — Open Embodied Foundation Models
- **类型：** repo（具身基础模型推理 + 权重入口）
- **机构：** 阿里巴巴达摩院（DAMO Academy, Alibaba Group）
- **链接：** https://github.com/alibaba-damo-academy/RynnBrain
- **项目页：** https://alibaba-damo-academy.github.io/RynnBrain
- **论文：** https://arxiv.org/abs/2607.17977（1.1）；1.0 见 arXiv:2602.14979 / 分支 `rynnbrain1.0`
- **Hugging Face：** https://huggingface.co/collections/Alibaba-DAMO-Academy/rynnbrain-11
- **ModelScope：** https://modelscope.cn/collections/DAMO_Academy/RynnBrain-11
- **Stars：** ~809（2026-07）
- **入库日期：** 2026-07-21
- **许可证：** Apache-2.0
- **代码 / 开源状态：** **部分开源** — **已发布** 基础模型 **2B / 9B / 122B-A10B** 权重与 **推理** 入口（`transformers`、`sglang`、`demo.py`、`cookbooks/`）；**未见** RynnBrain-VLA 训练/部署代码与 VLA 权重
- **一句话说明：** 达摩院开放具身基础模型族仓库：统一配方多尺度 checkpoint + 感知/定位/接触点/3D grounding cookbook；VLA 真机结果在论文与项目页，复现闭环需自建后训练栈。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-rynnbrain-1-1.md`](../../wiki/entities/paper-rynnbrain-1-1.md)
- **交叉归档：** [rynnbrain-alibaba-damo.md](../sites/rynnbrain-alibaba-damo.md)、[rynnbrain_1_1_arxiv_2607_17977.md](../papers/rynnbrain_1_1_arxiv_2607_17977.md)

---

## 仓内结构（2026-07 快照）

| 路径 | 作用 |
|------|------|
| `README.md` | 模型表、transformers / SGLang Quick Start、News |
| `demo.py` | Gradio 图/视频 + 视觉 prompt（object/area）推理 Demo |
| `cookbooks/` | `1_spatial_understanding` … `8_3d_grounding` 等 Jupyter；含 contact point / 3D |
| `RynnBrain_1_1.pdf` | 技术报告 PDF 镜像 |
| 分支 `rynnbrain1.0` | 1.0 资源入口 |

## Model Zoo（公开权重）

| 模型 | Base | HF |
|------|------|----|
| RynnBrain1.1-2B | Qwen3.5-2B | [Alibaba-DAMO-Academy/RynnBrain1.1-2B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-2B) |
| RynnBrain1.1-9B | Qwen3.5-9B | [Alibaba-DAMO-Academy/RynnBrain1.1-9B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-9B) |
| RynnBrain1.1-122B-A10B | Qwen3.5-122B-A10B | [Alibaba-DAMO-Academy/RynnBrain1.1-122B-A10B](https://huggingface.co/Alibaba-DAMO-Academy/RynnBrain1.1-122B-A10B) |

## 推理入口（README）

```bash
pip install transformers==5.2.0
# AutoModelForImageTextToText + AutoProcessor；model_path=Alibaba-DAMO-Academy/RynnBrain1.1-2B
# 或：python3 -m sglang.launch_server --model-path Alibaba-DAMO-Academy/RynnBrain1.1-2B --port 8000
```

---

## 对 wiki 的映射

- 实体页：[RynnBrain 1.1](../../wiki/entities/paper-rynnbrain-1-1.md)
- 方法交叉：[VLA](../../wiki/methods/vla.md)、[Action Chunking](../../wiki/methods/action-chunking.md)、[SONIC](../../wiki/methods/sonic-motion-tracking.md)
- 概念交叉：[Embodied Scaling Laws](../../wiki/concepts/embodied-scaling-laws.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md)
- 同院系对照：[RynnWorld-4D](../../wiki/entities/paper-rynnworld-4d-rgb-depth-flow.md)、[Qwen-VLA](../../wiki/entities/qwen-vla.md)
