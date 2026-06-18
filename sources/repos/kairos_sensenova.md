# kairos-agi/kairos-sensenova

> 来源归档

- **标题：** Kairos SenseNova（官方实现）
- **类型：** repo
- **组织：** kairos-agi
- **代码：** <https://github.com/kairos-agi/kairos-sensenova>
- **技术报告：** <https://arxiv.org/abs/2606.16533>
- **权重：** <https://huggingface.co/kairos-agi>；ModelScope：<https://modelscope.cn/collections/kairos-team/kairos30>
- **入库日期：** 2026-06-18
- **一句话说明：** Kairos **原生世界模型栈** 官方仓库入口：与 arXiv:2606.16533 技术报告配套的代码与 **Kairos 3.0** 权重发布渠道（HF / ModelScope）；实现 **理解–生成–预测统一 MoT**、**混合线性时序 DiT** 与 **WAM（Video DiT + Action DiT）** 部署路径。

## 与本仓库知识的关系

| 主题 | 关系 |
|------|------|
| [Kairos（原生世界模型栈）](../../wiki/entities/paper-kairos-native-world-model-stack.md) | 实体归纳页：CEDC、SWA/DSWA/GLA、WAM 与 benchmark |
| [Generative World Models](../../wiki/methods/generative-world-models.md) | **4B 级** 具身视频 WM + **线性可扩展** DiT 推理的对照样本 |
| [World Action Models](../../wiki/concepts/world-action-models.md) | **Joint WAM**：Video/Action 双 DiT + action-only 推理模式 |
| [Cosmos 3](../../wiki/entities/cosmos-3.md) | 同为 **Physical AI 世界模型平台**；Cosmos 3 偏 **全模态 16B/64B 开源栈**，Kairos 强调 **原生 CEDC 预训练 + GLA 长程记忆 + 边缘部署** |
| [HomeWorld](../../wiki/entities/paper-homeworld-whole-home-scene-generation.md) | **品牌名易混**：HomeWorld 为 **静态全屋 3D 场景生成**（Kairos-HomeWorld），与本仓库 **视频/WAM 世界模型** 无关 |

## 对 wiki 的映射

- 技术报告摘录：[`sources/papers/kairos_arxiv_2606_16533.md`](../papers/kairos_arxiv_2606_16533.md)
- 沉淀 **[`wiki/entities/paper-kairos-native-world-model-stack.md`](../../wiki/entities/paper-kairos-native-world-model-stack.md)**
