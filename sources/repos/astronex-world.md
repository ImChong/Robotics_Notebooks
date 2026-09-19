# Astronex-World

- **URL：** <https://github.com/Astronex-Robotics/Astronex-World>
- **论文：** [arXiv:2609.20034](https://arxiv.org/abs/2609.20034)
- **项目页：** <https://world.astronex.com.cn>
- **权重：** <https://huggingface.co/Astronex-Lab/Astronex-World>
- **许可：** Apache-2.0
- **开源：** 已开源（2026-09-19）

## 主要入口

- `inference/generate.py` — T2V/I2V 因果/双向采样，支持 event
- `post_train/train.py` — `--recipe camera | action | sft`
- `scripts/check_weights.py` — 校验 HF 权重目录完整性
