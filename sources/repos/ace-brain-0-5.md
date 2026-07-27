# ACE-Brain-0.5

> 来源归档

- **标题：** ACE-Brain-0.5 — Unified Embodied Foundation Model
- **类型：** repo（官方发布入口；当前以文档与权重导航为主）
- **机构：** ACE-Brain Team / 大晓机器人（Ace Robotics）
- **链接：** <https://github.com/ACE-BRAIN-Team/ACE-Brain-0.5>
- **组织镜像：** <https://github.com/DAXIAORobotics/ACE-Brain-0.5>（API `full_name` 指向此仓）
- **项目页：** <https://ace-brain-team.github.io/ACE-Brain-0.5/>
- **论文：** <https://arxiv.org/abs/2607.04426>
- **Hugging Face：** <https://huggingface.co/ACE-Brain/ACE-Brain-0.5-8B>
- **Stars：** ~36（2026-07）
- **入库日期：** 2026-07-27
- **许可证：** 仓内未声明 LICENSE 文件（截至核查日）
- **代码 / 开源状态：** **部分开源** — **已发布** 技术报告链接 + teaser/架构图资产 + **HF 8B 权重**与 transformers 推理样例；**未见** SSR+ 训练脚本、Action Expert / Fast Vision 训练代码、导航·操作闭环评测入口
- **一句话说明：** ACE-Brain-0.5 官方 GitHub 落地页：读论文与下载权重的入口；可复现推理走 Hugging Face，端到端训练栈仍待发布。
- **沉淀到 wiki：** 是 → [`wiki/entities/paper-ace-brain-0-5.md`](../../wiki/entities/paper-ace-brain-0-5.md)
- **交叉归档：** [ace-brain-0-5-github-io.md](../sites/ace-brain-0-5-github-io.md)、[ace_brain_0_5_arxiv_2607_04426.md](../papers/ace_brain_0_5_arxiv_2607_04426.md)

---

## 仓内结构（2026-07-27 快照）

| 路径 | 作用 |
|------|------|
| `README.md` | 简介、SSR+、能力评测摘要、bibtex |
| `assets/` | logo / teaser / architecture 图 |

## Hugging Face 推理入口（模型卡）

```python
from transformers import Qwen3VLForConditionalGeneration, AutoProcessor

model = Qwen3VLForConditionalGeneration.from_pretrained(
    "ACE-Brain/ACE-Brain-0.5-8B",
    dtype="auto",
    device_map="auto",
)
processor = AutoProcessor.from_pretrained("ACE-Brain/ACE-Brain-0.5-8B")
# apply_chat_template → generate → batch_decode
```

模型类型：`qwen3_vl` / `Qwen3VLForConditionalGeneration`（约 8B VLM 权重分片 `model-0000*-of-00004.safetensors`）。

---

## 对 wiki 的映射

- 实体页：[ACE-Brain-0.5](../../wiki/entities/paper-ace-brain-0-5.md)
- 方法交叉：[VLA](../../wiki/methods/vla.md)、[Foundation Policy](../../wiki/concepts/foundation-policy.md)
- 概念交叉：[过程奖励建模](../../wiki/concepts/progress-reward-modeling.md)
- 同机构对照：[Kairos](../../wiki/entities/paper-kairos-native-world-model-stack.md)
