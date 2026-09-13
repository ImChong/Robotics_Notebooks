# PhysBrain 1.5 项目页（deepcybo-physai.github.io/PhysBrain-1.5）

> 来源归档

- **标题：** PhysBrain 1.5 — From General VLMs to Physical Foundation Model
- **类型：** site / project-page
- **URL：** <https://deepcybo-physai.github.io/PhysBrain-1.5/>
- **技术报告：** <https://github.com/DeepCybo-PhysAI/PhysBrain-1.5/blob/main/tech_report.pdf>
- **论文（前作 / 学术脉络）：** <https://arxiv.org/abs/2512.16793>
- **代码 / 文档仓：** <https://github.com/DeepCybo-PhysAI/PhysBrain-1.5>
- **评测工具：** <https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit>
- **权重集合：** <https://huggingface.co/collections/DeepCybo/physbrain-15>
- **在线 Demo：** <https://huggingface.co/spaces/hugging-apps/physbrain1-5-8b-demo>
- **机构：** 机智赛博（DeepCybo）；北京中关村学院（Zhongguancun Academy）；中关村人工智能研究院（ZGCI / ZGCA）
- **入库日期：** 2026-09-13
- **一句话说明：** 官方项目站：统一具身理解 / 动作生成 / 未来状态预测叙事、28 项 benchmark 榜单与定性示例；导航链到技术报告、HF 权重与 PhysBrainEvalKit。

## 开源核查（步骤 2.5，截至 2026-09-13）

| 核查项 | 结论 |
|--------|------|
| 项目页是否链到权重 | 是 → Hugging Face `DeepCybo/PhysBrain1.5-8B` / `PhysBrain1.5-2B` |
| 项目页是否链到评测代码 | 是 → GitHub `DeepCybo-PhysAI/PhysBrainEvalKit` |
| 项目页是否链到文档仓 | 是 → GitHub `DeepCybo-PhysAI/PhysBrain-1.5`（技术报告 + README） |
| 仓内可运行训练入口 | **否** — `PhysBrain-1.5` 仓仅技术报告与项目文档 |
| HF 可运行推理 | **是** — 2B / 8B 权重 + 官方 Demo Space |
| EvalKit 可运行评测 | **是** — 28 benchmark 适配器 + 分片 runner |
| 综合判定 | **部分开源**（权重 + 评测工具 + Demo 已发布；预训练 / SFT 训练栈未见） |

## 公开信息要点

- Hero：**72.5** overall（8B，28 benchmark 未加权均值）、开源榜 **#1**（14 项第一、10 项第二）；骨干 **Qwen3-VL**。
- 三能力闭环：**Embodied understanding** / **Action generation（ActionPiece tokens）** / **Future-state prediction（RGB + depth + robot mask）**；统一 next-token 预测，无任务专用 head。
- 数据：预训练监督来自人类交互视频（ego / ego–exo / panoramic）；SFT 混合人类示范、真机轨迹与仿真经验。
- 部署：宣称兼容 Transformers / vLLM / SGLang / LLaMA-Factory / ms-swift / veRL 等标准接口。

## 关联资料

- 技术报告摘录：[`sources/papers/physbrain_1_5_technical_report_2026.md`](../papers/physbrain_1_5_technical_report_2026.md)
- 文档仓归档：[`sources/repos/physbrain-1-5.md`](../repos/physbrain-1-5.md)
- 评测仓归档：[`sources/repos/physbrain-eval-kit.md`](../repos/physbrain-eval-kit.md)
- 前作 arXiv 策展摘录：[`sources/papers/sun_awesome_ego_2512_16793_physbrain-human-egocentric-data-as-a-bri.md`](../papers/sun_awesome_ego_2512_16793_physbrain-human-egocentric-data-as-a-bri.md)
- Wiki 实体：[`wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md`](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)
