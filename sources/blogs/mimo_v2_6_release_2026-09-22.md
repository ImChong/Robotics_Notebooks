# MiMo-V2.6 发布说明（小米 MiMo 官网）

> 来源归档

- **标题：** MiMo-V2.6：扩展强化学习规模，迈向自我提升
- **类型：** blog / news（官方发布说明）
- **链接：** <https://mimo.mi.com/docs/zh-CN/news/latest/v2-6>
- **英文博客：** <https://mimo.xiaomi.com/mimo-v2-6>（模型卡亦链此址）
- **更新日期：** 2026-09-22（页内标注）
- **入库日期：** 2026-09-23
- **一句话说明：** 正式发布并开源 **MiMo-V2.6-Pro / Flash** 原生全模态 MoE；强调 **RSI 路径**、Live RL **6 天**训练叙事、**Vibe World**（3D 游戏 / Blender / **具身仿真 Franka** / CUA）与科研 Co-Scientist 案例；API 定价沿用 V2.5。

## 开源核查（步骤 2.5）

| 项 | 结论 |
|----|------|
| **模型权重** | **已开源** — [HF 集合 mimo-v26](https://huggingface.co/collections/XiaomiMiMo/mimo-v26) |
| **技术报告** | **已公开** — PDF 随 Pro-RL 仓 |
| **RL 代码与环境** | **已开源** — 7k+ 环境、verl / uni-agent、mini-harnesses（详见发布说明「全面开源」节） |
| **API / Desktop** | 闭源产品 — [开放平台](https://platform.xiaomimimo.com) · [Desktop](https://mimo.xiaomimimo.com/desktop/) |

## 核心发布要点（2026-09-22 摘录）

### 定位与 benchmark 叙事

- **RSI（递归自我改进）：** 以可验证复杂任务为基础，规模化 RL 算力。
- **AA Intelligence Index：** Pro **46 分**，称当前最强开源；仍落后 Claude Fable 5.1 / GPT-6 Astra 等闭源（页内表述）。
- **性价比：** 同等智能下价格为海外模型 **1/20–1/60**；API 定价与 V2.5 相同。

### Live RL 训练（公开实验）

| 模型 | 训练成本（页内） | 步数 | 轨迹量 | DeepSWE v1.1 提升 |
|------|------------------|------|--------|-------------------|
| Flash | ~**85 万美元** | 30 | （合计 ~75 万） | 48.8 → **65.7**（+17） |
| Pro | ~**262 万美元** | 30 | | 58.4 → **72.6**（+14） |

任务平均通过率：Flash **+25%**、Pro **+12%**。

### RL 三维扩展（与报告一致）

1. 更大 Batch / 更高吞吐（1,568 样本、1M 上下文、单步 **3.5–3.7B token**）
2. 更多任务（Code / General / Visual / Cyber + 多 Harness 混合）
3. 更大 Grader 算力（组内相对比较 → 自我改进闭环）

工程措施：冻结 MoE Router、Reward Hacking 防线、统一轨迹表示、控制面/数据面解耦。

### Vibe World（与机器人相关摘录）

- **具身智能：** 多视角相机输入 → 持续推理决策 → 视觉反馈闭环控制 **Franka Panda**（抓取、颜色匹配、精准放置）。
- **3D / Blender / CUA：** 游戏世界搭建、Blender 资产、Computer Use Agent。

### 全面开源清单（页内）

- **7k+ RL 任务环境**（软件工程、漏洞复现、知识型工作、网页设计等）
- **端到端 RL 框架**（verl、uni-agent、mini-swe-agent）
- **mini-harnesses** + Multi-Harness Training
- **MiMo-V2.6-Distill-Qwen-9B** 及 Distill 后 RL 提升数据（11 项评测均优于 SFT 基线）

开源入口：<https://huggingface.co/collections/XiaomiMiMo/mimo-v26>

### API 模型名（页内注意）

调用时使用全小写：`mimo-v2.6-pro`、`mimo-v2.6-flash`、`mimo-v2.6-pro-ultraspeed`。

## 对 wiki 的映射

- 主实体：[MiMo-V2.6](../../wiki/entities/mimo-v2-6.md)
- 技术报告：[mimo_v2_6_technical_report_2026.md](../papers/mimo_v2_6_technical_report_2026.md)
- 仓库索引：[mimo-v2-6.md](../repos/mimo-v2-6.md)
