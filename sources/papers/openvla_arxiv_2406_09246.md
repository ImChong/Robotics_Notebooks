# OpenVLA: An Open-Source Vision-Language-Action Model（arXiv:2406.09246）

> 来源归档（ingest）

- **标题：** OpenVLA: An Open-Source Vision-Language-Action Model
- **短名：** OpenVLA
- **类型：** paper
- **arXiv：** <https://arxiv.org/abs/2406.09246>
- **PDF：** <https://arxiv.org/pdf/2406.09246>
- **项目页：** <https://openvla.github.io/>
- **代码：** <https://github.com/openvla/openvla>
- **机构：** 斯坦福大学；加州大学伯克利分校；马克斯·普朗克研究所
- **入库日期：** 2026-09-02
- **索引来源：** [具身智能研究室 VLA/WM 阅读路线](../blogs/wechat_embodied_ai_lab_vla_wm_reading_roadmap_2026-09-02.md)
- **一句话说明：** 7B 开源 VLA；DINOv2+SigLIP 双塔 + Llama 2；OXE 预训练；LoRA/OFT 微调。

## 开源状态（步骤 2.5，2026-09-02）

- **已开源**：`openvla/openvla` + Hugging Face `openvla/openvla-7b` 权重。仓库实体见 [openvla](../repos/openvla.md)。

## 核心摘录（面向 wiki 编译）

- 视觉：DINOv2（几何）+ SigLIP（语言对齐）；语言主干 Llama 2 7B；7 维动作各 256 bin 自回归。
- 数据：Open X-Embodiment（约 970k 轨迹）+ 内部数据；目标机器人微调 5k–10k 步即可适配。
- **对 wiki 的映射：** [paper-openvla](../../wiki/entities/paper-openvla.md)；软件实体 [openvla](../../wiki/entities/openvla.md)

## 当前提炼状态

- [x] 项目页/仓库已交叉核查
- [x] wiki 映射：`wiki/entities/paper-openvla.md` 新建（不与 `openvla.md` 软件实体重复 arXiv）
