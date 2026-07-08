# LingBot-VLA 2.0

> 来源归档

- **标题：** LingBot-VLA 2.0 — From Foundation to Application
- **类型：** repo
- **组织：** Robbyant（蚂蚁集团具身智能）
- **代码：** <https://github.com/robbyant/lingbot-vla-v2>
- **项目页：** <https://technology.robbyant.com/lingbot-vla-v2>
- **技术报告：** <https://arxiv.org/abs/2607.06403>（PDF 见仓库 `assets/LingBot_VLA_2_0.pdf`）
- **权重：** <https://huggingface.co/robbyant/lingbot-vla-v2-6b> / <https://modelscope.cn/models/Robbyant/lingbot-vla-v2-6b>
- **依赖骨干：** [Qwen3-VL-4B-Instruct](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct)、[MoGe-2-vitb-normal](https://huggingface.co/Ruicheng/moge-2-vitb-normal)、LingBot-Depth / DINO-VIDEO 教师（见 HF 集合子目录）
- **入库日期：** 2026-07-08
- **一句话说明：** LingBot-VLA 2.0 官方实现：6B 预训练权重、异构数据过滤管线、55 维统一动作 + MoE expert、Dual-Query 深度/视频蒸馏、RoboTwin 后训练范例与真机 `deploy.lingbot_vla_v2_policy` 部署入口。
- **沉淀到 wiki：** [LingBot-VLA 2.0](../../wiki/entities/lingbot-vla-v2.md)

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [VLA](../../wiki/methods/vla.md) | **Qwen3-VL + MoE action expert + flow 去噪**；相对 1.0 扩展 **全身 DoF** 与 **预测性蒸馏** |
| [HumanNet](../../wiki/entities/humannet.md) | 同类 **egocentric 人视频小时** 作 VLA 持续预训练来源的实证语境（LingBot 系受控对比） |
| [Manipulation](../../wiki/tasks/manipulation.md) | GM-100 / RoboTwin 双臂 generalist 后训练与评测 |
| [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) | Astribot / 移动底盘长程家务类任务评测 |

## 为何值得保留

- 技术报告 + **开源 6B 权重** + **完整后训练/部署脚本**，是 2026 年 **务实 VLA 基础模型 → 真机应用** 路线的直接复现入口。
- **55 维统一动作** 与 **MoE 跨本体 scaling** 可与 Green-VLA、Qwen-VLA 等 **语义槽位 / embodiment prompt** 路线形成工程对照。
- 与 [Humanoid_Robot_Learning_Paper_Notebooks](https://github.com/ImChong/Humanoid_Robot_Learning_Paper_Notebooks) 单篇深读互补：本库负责 **跨主题 VLA 数据–模型–部署** 索引。
