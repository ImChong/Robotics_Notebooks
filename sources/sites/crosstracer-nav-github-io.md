# CrossTracer 项目页（lilduckkk.github.io/CrossTracer-Nav）

> 来源归档（ingest 配套站点）

- **URL：** <https://lilduckkk.github.io/CrossTracer-Nav/>
- **标题：** CrossTracer: Cross-Embodiment Navigation via VLA Model Reasoning and Trace Residuals Adapting
- **机构：** 鹏城实验室；南方科技大学；创新投资研究院；苏州大学
- **论文：** <https://arxiv.org/abs/2608.06688> — 归档见 [`sources/papers/crosstracer_arxiv_2608_06688.md`](../papers/crosstracer_arxiv_2608_06688.md)
- **入库日期：** 2026-09-04
- **一句话说明：** CrossTracer 落地页：动机、两段框架（VL-Tracer / CE-Adapter / CE-RRT*）、NaviTrace 分数与轮式/腿式语言导航叙事。截至入库日 **无 Code / GitHub / Hugging Face 链接**。

## 开源核查（步骤 2.5，2026-09-04）

| 项 | 状态 |
|----|------|
| **论文承诺** | 正文只写 *Experiment videos and more details can be found at* 项目页；未写 "code will be released" 的明确日期。NaviTrace 对照表把 CrossTracer-8B 标 **Open-Source ✓**（该列同时区分 Qwen 开源 vs Gemini/Claude 闭源）。 |
| **项目页 Code 区** | **无** GitHub、Hugging Face、Zenodo、ModelScope。页面为摘要 + Approach + Experiments 叙事；页脚注明站点模板改编自 Nerfies。 |
| **作者公开仓** | GitHub [`Lilduckkk`](https://github.com/Lilduckkk) 截至核查日仅有 `whisper_turtlesim`，**无** CrossTracer / CrossTracer-Nav 仓。 |
| **结论** | **宣称开源 / 待核实**。对照表打勾 ≠ 项目页给出可跑入口。勿写「已开源」；勿建 `sources/repos/`。放出训练/推理仓后再补仓库归档与论文页时序图。 |

## 页面结构速记

1. **Motivation** — 语义目标 vs 本体可通行性：腿式可越粗糙地面/小高差，轮式需绕行；经典局部规划缺开放词汇。
2. **CrossTracer** — 像素轨迹作共享接口；机器人 ID 只进 refinement，避免 proposer 把语义与本体缠死；推理只要 RGB + 目标 + 本体 ID。
3. **CE-Adapter** — FiLM 调视觉特征；轨迹 query 交叉注意力；残差头有界；训练另有 Feasibility / Sensitivity 头。
4. **NaviTrace** — 总分 45.68；相对 Gemini-2.5-Pro +28%；去 CE-Adapter → 22.56；加 goal-pose → 63.91。
5. **真机** — 轮式 SR 0.40→0.65、SPL 0.37→0.59、STT 0.17→0.30；腿式 SR 0.45→0.70、SPL 0.31→0.58、STT 0.27→0.43（相对 OmniVLA）。

## 关联资料

- 论文摘录：[`sources/papers/crosstracer_arxiv_2608_06688.md`](../papers/crosstracer_arxiv_2608_06688.md)
- Wiki 实体：[`wiki/entities/paper-crosstracer.md`](../../wiki/entities/paper-crosstracer.md)
- 评测基准（外部）：NaviTrace — <https://arxiv.org/abs/2510.26909> / <https://leggedrobotics.github.io/navitrace_webpage/>
- 提案骨干（外部）：OmniVLA — <https://arxiv.org/abs/2509.19480>
