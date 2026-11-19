# Behavior Prompting Policy 项目页

- **论文：** [arXiv:2606.30457](https://arxiv.org/abs/2606.30457)
- **项目页：** <https://behavior-prompting.github.io/>
- **代码：** <https://github.com/real-stanford/behavior_prompting> / <https://github.com/real-stanford/iPhUMI>
- **机构：** Stanford University; UC Berkeley
- **入库日期：** 2026-11-19
- **核对范围：** 通读官方项目页全文（视频展示逻辑、方法摘要、Prompt vs Goal-Image / Language / Finetuned VLA 对比、DrawAnything / LIBERO-Gen / Laundry Folding 互动演示说明、BibTeX），并与仓库文档 `docs/` 核对开放边界。
- **通读：** 已完成

## 开源与公开资源核查（2026-11-19）

| 组件 | 状态 | 说明 |
|------|------|------|
| 项目页 | 有 | 展示 iPhUMI 单次手持示教作为 prompt 的真机绘画/折衣、LIBERO-Gen 与实时提示推理演示 |
| 核心代码 | **已开源** | [real-stanford/behavior_prompting](https://github.com/real-stanford/behavior_prompting)（MIT） |
| 手持采集仓 | **已开源** | [real-stanford/iPhUMI](https://github.com/real-stanford/iPhUMI) |
| 数据与权重 | **已发布** | Hugging Face `austinpatel/*` 涵盖 DrawAnything Sim/Real、LIBERO、LIBERO-Gen、Laundry Folding 与 `pi05_*_lora` 对照权重 |
| 未发布部分 | **部分保留** | 仓内标注无线 iPhUMI 实时传 prompt 接口未释放；`DrawAnything-Real` 的机械臂程序化生成与 Chamfer 自动评测脚本未公开 |

**结论：** **已开源**（策略训练/评估/离线 prompt 部署 + 数据/Checkpoint 完整可取，无线实时端到端 iPhUMI 推送链路与真机自动绘测脚本未释出）。

## 交叉

- 论文归档：[`sources/papers/behavior_prompting_arxiv_2606_30457.md`](../papers/behavior_prompting_arxiv_2606_30457.md)
- 仓库归档：[`sources/repos/real-stanford-behavior-prompting.md`](../repos/real-stanford-behavior-prompting.md)
- wiki：[`wiki/entities/paper-behavior-prompting-policy.md`](../../wiki/entities/paper-behavior-prompting-policy.md)
