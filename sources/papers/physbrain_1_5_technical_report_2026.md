# PhysBrain 1.5 Technical Report

> 来源归档

- **标题：** PhysBrain 1.5: From General VLMs to Physical Foundation Model
- **类型：** technical-report / project-preprint
- **PDF：** <https://github.com/DeepCybo-PhysAI/PhysBrain-1.5/blob/main/tech_report.pdf>
- **项目页：** <https://deepcybo-physai.github.io/PhysBrain-1.5/>
- **前作 arXiv：** <https://arxiv.org/abs/2512.16793> — *PhysBrain: Human Egocentric Data as a Bridge from Vision Language Models to Physical Intelligence*
- **机构：** 机智赛博（DeepCybo）；北京中关村学院（Zhongguancun Academy）；中关村人工智能研究院（ZGCI / ZGCA）
- **入库日期：** 2026-09-13
- **代码：** <https://github.com/DeepCybo-PhysAI/PhysBrain-1.5>（文档仓）；评测 <https://github.com/DeepCybo-PhysAI/PhysBrainEvalKit>
- **权重：** <https://huggingface.co/collections/DeepCybo/physbrain-15>
- **一句话说明：** PhysBrain 1.5 技术报告：在 Qwen3-VL 上扩展 action / visual-state token，用统一自回归目标同时学具身理解、ActionPiece 动作块与未来 RGB+depth+mask 预测；28 benchmark 开源 SOTA。

---

## 核心摘录

1. **统一物理闭环：** 观测 → 推理与动作 → 环境变化 → 新观测；语言、空间输出、末端轨迹与未来视觉状态均离散化为 token，共享单一自回归骨干，**无任务专用输出头**。
2. **三能力一体：** （1）具身理解：视觉–空间感知、3D/多视角、规划、指向与 affordance、视觉轨迹推理；（2）动作生成：ActionPiece token + 跨本体统一 action codebook；（3）未来状态：空间对齐的 RGB、深度与机器人 mask。
3. **数据配方：** 预训练监督全部来自人类交互视频（ego、同步 ego–exo、全景，按任务中心 episode 组织）；SFT 混合人类示范、真机轨迹与仿真经验。
4. **骨干与规模：** 基于预训练 **Qwen3-VL**；发布 **2B** 与 **8B** 两档 checkpoint。
5. **28 benchmark 结果（技术报告 / 项目页，独立重评、每榜单一指标）：** PhysBrain 1.5-8B **Overall 72.5**（0–100 未加权均值），开源模型中 **14 项第一、10 项第二**；2B 为 **66.6**（参考，不参与开源排名）。闭源参照：GPT-6-Astra **73.3**、Gemini 3.6 Flash **73.0**。
6. **部署接口：** 宣称兼容 Transformers / vLLM / SGLang / LLaMA-Factory / ms-swift / veRL 等标准 VLM 推理与后训练栈。

## 开源边界（步骤 2.5）

| 已发布 | 备注 |
|--------|------|
| 技术报告 PDF | `PhysBrain-1.5` 仓 |
| HF 权重 2B / 8B | `DeepCybo/PhysBrain1.5-*` |
| 在线 Demo | HF Space `hugging-apps/physbrain1-5-8b-demo` |
| 28 benchmark 评测 | `PhysBrainEvalKit` 完整 runner |
| 预训练 / SFT 训练代码 | **未见** |
| 人类交互视频数据 | **未见** 公开下载入口 |

## 对 wiki 的映射

- 实体页：[`wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md`](../../wiki/entities/paper-sa-2512-16793-physbrain-human-egocentric-data-as-a-bridge-from.md)
- 项目页：[`sources/sites/physbrain-1-5-github-io.md`](../sites/physbrain-1-5-github-io.md)
- 仓库：[`sources/repos/physbrain-1-5.md`](../repos/physbrain-1-5.md)、[`sources/repos/physbrain-eval-kit.md`](../repos/physbrain-eval-kit.md)
- 前作策展：[`sources/papers/sun_awesome_ego_2512_16793_physbrain-human-egocentric-data-as-a-bri.md`](./sun_awesome_ego_2512_16793_physbrain-human-egocentric-data-as-a-bri.md)
