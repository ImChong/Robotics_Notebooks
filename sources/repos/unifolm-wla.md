# unifolm-wla

> 来源归档

- **标题：** unifolm-wla
- **类型：** repo
- **来源：** unitreerobotics（Unitree 官方 GitHub 组织）
- **链接：** https://github.com/unitreerobotics/unifolm-wla
- **项目页：** https://unigen-x.github.io/unifolm-wla.github.io/
- **星标（截至 2026-09-18）：** ~142
- **创建 / 最近推送（2026-09-18 快照）：** 2026-09-11 / 2026-09-16
- **主要语言：** Markdown / 技术文档（截至入库日无 Python 训练推理入口）
- **分类：** 基础模型（UnifoLM）
- **入库日期：** 2026-09-18
- **一句话说明（2026-09-18 快照）：** 官方 UnifoLM-WLA-1.0 枢纽仓：6B 通用人形 WLA 模型说明、HF 权重/数据集索引与统一动作空间处理规范；ER-1/ER-Flow 权重已发，WLA-Base 与后训练代码待发布。
- **沉淀到 wiki：** 是 → [`wiki/entities/unifolm-wla.md`](../../wiki/entities/unifolm-wla.md)
- **组织地图：** [`sources/repos/unitree.md`](unitree.md)

---

## README 要点（编译自上游）

- Sep 11, 2026: 发布 [UnifoLM-ER-1](https://huggingface.co/unitreerobotics/UnifoLM-ER-1) 与 [UnifoLM-ER-Flow](https://huggingface.co/unitreerobotics/UnifoLM-ER-Flow) 权重。
- 6B 参数；约 2,500 小时真机数据；单模型 64 任务（桌面 + 全身）；适配二指夹爪与多种五指灵巧手。
- 技术文档：[`docs/robot_action_state_processing_en.md`](https://github.com/unitreerobotics/unifolm-wla/blob/main/docs/robot_action_state_processing_en.md) — 统一 54 维动作 / 60 维状态、SE(3) 相对位姿、分模块归一化与 mask 规范。

## Open-Source Plan（2026-09-18）

| 类别 | 项 | 状态 |
|------|-----|------|
| 代码 | Post-Train Code | 未发布 |
| 模型 | UnifoLM-ER-1 | 已发布 |
| 模型 | UnifoLM-ER-Flow | 已发布 |
| 模型 | UnifoLM-WLA-Base | 待发布 |
| 数据 | UniBot-V1 Challenge Dataset | 已发布 |
| 数据 | Unitree-WBT-Dataset | 待发布 |
| 数据 | Unitree-Manipulation-Dataset | 待发布 |

## 开源状态

- **部分开源**：公开 GitHub 仓库与 HF 中间权重/部分数据集；**无可运行**训练/推理/部署脚本（截至 2026-09-18）。

## 对 wiki 的映射

- 实体页：[`wiki/entities/unifolm-wla.md`](../../wiki/entities/unifolm-wla.md)
- 项目页：[`sources/sites/unifolm-wla-github-io.md`](../sites/unifolm-wla-github-io.md)
- 组织枢纽：[`wiki/entities/unitree.md`](../../wiki/entities/unitree.md)

## 后续发布核对（2026-10-06）

- **核对版本：** [`0a1aa87`](https://github.com/unitreerobotics/unifolm-wla/tree/0a1aa87be8f775c7883fcd27216e3666ac18a8c5)；上面的 2026-09-18 表是历史快照，不代表当前状态。
- **2026-09-11：** ER-1 / ER-Flow 权重开放。
- **2026-09-20：** 模型模块与动作专家训练代码开放。
- **2026-09-28：** WLA-1.0-Base 权重与微调代码开放。
- **日期依据：** [官方 README News](https://github.com/unitreerobotics/unifolm-wla/blob/0a1aa87be8f775c7883fcd27216e3666ac18a8c5/README.md#-news)，不使用仓库创建日或入库日替代各资产发布日。

### 当前开放边界

训练、微调、LoRA、离线评估与模型服务已有代码入口；[WLA-1.0-Base 权重](https://huggingface.co/unitreerobotics/UnifoLM-WLA-1.0-Base)、[WBT 数据集合](https://huggingface.co/collections/unitreerobotics/unifolm-wbt-dataset) 与 [Dex1 数据集合](https://huggingface.co/collections/unitreerobotics/unifolm-g1-dex1-dataset) 已列公开资源。数据集合可访问不意味着约 2,500h 训练池及全部 64 任务配方完整开放；数据集具体开放日期本次未确认。

[官方训练/评估文档](https://github.com/unitreerobotics/unifolm-wla/blob/0a1aa87be8f775c7883fcd27216e3666ac18a8c5/docs/train_action_expert_en.md) 提供 `uv sync`、`flash-attn` 安装、`examples/unifolm_wla/eval_files/unitree/eval_local_episode.py` 和 `model_server/action_server_wbc_msgpack_unitree.py`。模型服务器现有协议仅覆盖 Dex1；WBT 需补手指与基座通道，不能把服务器入口视为全部任务已可直接部署。
