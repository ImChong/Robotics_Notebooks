# AlphaBrain

> 来源归档

- **标题：** AlphaBrain
- **类型：** repo
- **来源：** AlphaBrain Team（NeuroVLA 等脑启发 VLA 维护方）
- **链接：** <https://github.com/AlphaBrainGroup/AlphaBrain>
- **文档：** <https://alphabraingroup.github.io/AlphaBrain/>
- **论文关联：** [NeuroVLA arXiv:2601.14628](../papers/neurovla_arxiv_2601_14628.md)
- **许可：** MIT
- **入库日期：** 2026-09-16
- **一句话说明：** 开源具身智能 **一体化框架**：多 VLA 架构（OFT / GR00T / PI / **NeuroVLA**）、世界模型骨干、持续学习与 RL-Token 训练范式共享同一 trainer/config/推理接口。
- **沉淀到 wiki：** [`wiki/entities/paper-neurovla.md`](../../wiki/entities/paper-neurovla.md)

---

## 核心定位

社区向「开箱即用」具身研究栈：Baseline VLA 微调评测、**NeuroVLA（脉冲动作头 + R-STDP）**、RL-Token、世界模型与持续学习模块化组合。

---

## NeuroVLA 相关入口

| 路径 | 说明 |
|------|------|
| `docs/quickstart/neurovla.md` | NeuroVLA 快速上手 |
| `scripts/run_brain_inspired_scripts/` | 预训练/微调脚本 |
| `deployment/model_server/server_policy.py` | 策略服务部署 |
| [HF AlphaBrainGroup](https://huggingface.co/AlphaBrainGroup) | 预训练权重（如 `neurovla-libero-all4suite`） |

---

## 开源结论（2026-09-16）

**已开源** — 训练、评测、部署脚本与权重可获取；论文自研 FPGA 神经形态硬件为附加部署选项，非复现必需。
