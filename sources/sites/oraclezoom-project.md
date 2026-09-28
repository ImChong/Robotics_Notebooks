# OracleZoom 官方项目页（dipta007）

> 来源归档

- **标题：** OracleZoom: Reference-Constrained Recursive Image Super-Resolution — Project Page
- **类型：** site / project-page
- **URL：** <https://dipta007.github.io/OracleZoom/>
- **关联论文：** <https://arxiv.org/abs/2609.06490>
- **关联代码：** <https://github.com/dipta007/OracleZoom>
- **权重 / Demo：** <https://huggingface.co/dipta007/OracleZoom> · <https://huggingface.co/spaces/dipta007/OracleZoom>
- **机构：** University of Maryland, Baltimore County（UMBC）
- **入库日期：** 2026-09-28
- **一句话说明：** **递归图像超分（Recursive SR）** 在极深倍率（至 **256×**）下 GT 不可得的 **监督缺口** 问题；OracleZoom 用 **on-policy 自蒸馏** + **最后一档可用 GT 作跨尺度参考**，配合无参考质量、KL 先验与 EMA 一致性，相对 Chain-of-Zoom 降低幻觉。

## 开源状态（项目页 / README 核查 2026-09-28）

- **已开源：** GitHub [`dipta007/OracleZoom`](https://github.com/dipta007/OracleZoom)（MIT；`uv sync` 单卡推理/训练）；HF 权重 [`dipta007/OracleZoom`](https://huggingface.co/dipta007/OracleZoom)（merged transformer + CoZ 依赖 ckpt）；训练数据 [`dipta007/OracleZoom-4KLSDB-train`](https://huggingface.co/datasets/dipta007/OracleZoom-4KLSDB-train)；在线 Demo [`spaces/dipta007/OracleZoom`](https://huggingface.co/spaces/dipta007/OracleZoom)。
- **依赖链：** 基于 [Chain-of-Zoom](https://github.com/bryanswkim/Chain-of-Zoom) 递归 zoom 与 VLM prompter、[OSEDiff](https://github.com/cswry/OSEDiff) 一步 SR 骨干；SD3-medium 与 Qwen2.5-VL-3B 首次运行自动拉取（SD3 需在 HF 接受许可）。
- **互指：** [`sources/papers/oraclezoom_arxiv_2609_06490.md`](../papers/oraclezoom_arxiv_2609_06490.md) · [`sources/repos/oraclezoom.md`](../repos/oraclezoom.md)

## 页面结构归纳

1. **叙事：** GT 在递归 zoom 结束前耗尽；OracleZoom 在 **无更深 GT** 时仍用 **最后一档可观测 GT** 约束可验证内容，其余细节在 KL 约束先验下由质量目标引导。
2. **规模：** rank-16 LoRA **7.1M** 可训参数，**1,000** 张图训练；发布为 **merged** transformer（非独立 LoRA 包）。
3. **定量（与 README 一致）：** 七测试集 CLIPIQA 均值 **0.713**；4× LPIPS **0.199**、DISTS **0.160**；256× CLIPIQA **0.706**；InternVL3.5-38B 在 64× / 256× **68% / 78%** 偏好 OracleZoom（相对 CoZ）；幻觉率 0.21 / 0.14 vs CoZ 0.55 / 0.70。
4. **资源链接：** Paper、Demo Space、Model、Dataset、HF Collection、GitHub。

## 对 wiki 的映射

- 沉淀：[`wiki/entities/paper-oraclezoom.md`](../../wiki/entities/paper-oraclezoom.md)
