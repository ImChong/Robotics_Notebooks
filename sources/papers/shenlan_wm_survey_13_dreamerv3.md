# DreamerV3（Mastering Diverse Domains through World Models，arXiv:2301.04104）

> 来源归档（ingest · 深蓝世界模型 15 项目 第 13/15 + 物理保真度博客交叉）

- **标题：** Mastering Diverse Domains through World Models（DreamerV3）
- **类型：** paper / DreamerV3 / world models / latent imagination / model-based RL
- **路线分类：** 03 虚拟沙盒（[深蓝 15 开源专题](../blogs/wechat_shenlan_world_models_15_open_source_2026.md)）
- **出处：** Nature（期刊版）；预印本 arXiv:2301.04104
- **论文链接：** <https://arxiv.org/abs/2301.04104>
- **作者页：** <https://danijar.com/dreamerv3>
- **代码（公开复现）：** <https://github.com/danijar/dreamerv3>（MIT）
- **后继开源：** [Open Dreamer](../repos/open-dreamer.md)（Dreamer 4 JAX 管线）
- **作者：** Danijar Hafner、Jurgis Pašukonis、Jimmy Ba、Timothy Lillicrap
- **机构：** 谷歌 DeepMind（Google DeepMind）等
- **文内引用（2026-06-02，深蓝策展）：** 1475
- **入库日期：** 2026-06-03；**加厚修订：** 2026-07-27
- **一句话说明：** 学习环境世界模型，在潜空间想象轨迹上训 actor-critic；**单一超参配置**在 **150+** 任务上超越大量专用方法；Minecraft 从零采钻石为标志性结果。

## 开源状态（2026-07-27）

- **已开源（社区权威复现）：** [danijar/dreamerv3](https://github.com/danijar/dreamerv3) · MIT · JAX；README 写明基于 DreamerV2 开源基座的 reimplementation，**与 Google/DeepMind 内部实现无关**。
- **后继：** Dreamer 4 见 [open-dreamer](../repos/open-dreamer.md)（训练管线已开、完整 agent 环仍待齐）。

## 核心摘录

- **RSSM 世界模型：** 感官 → 离散/分类表示；给定动作预测未来表示与回报。
- **想象学习：** 在模型生成的 latent 轨迹上优化策略，减少真环境步数。
- **稳健技巧：** 归一化、损失平衡、变换等，使同一配置跨域稳定。
- **物理保真度读法：** 属「低维潜在状态」输出族（见微信策展）；快但压缩可能丢掉接触细节。

## 对 wiki 的映射

- Wiki：[`wiki/entities/paper-shenlan-wm-13-dreamerv3.md`](../../wiki/entities/paper-shenlan-wm-13-dreamerv3.md)
- 代码：[`sources/repos/danijar-dreamerv3.md`](../repos/danijar-dreamerv3.md)
- 后继：[`sources/repos/open-dreamer.md`](../repos/open-dreamer.md)
- [world-models-15-open-source-technology-map](../../wiki/overview/world-models-15-open-source-technology-map.md)
- [world-models-route-03-virtual-sandbox](../../wiki/overview/world-models-route-03-virtual-sandbox.md)
- [物理保真度博客](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)

## 参考来源（原始）

- 论文：<https://arxiv.org/abs/2301.04104>
- 微信公众号编译：[wechat_shenlan_world_models_15_open_source_2026.md](../blogs/wechat_shenlan_world_models_15_open_source_2026.md)
- 物理保真度策展：[wechat_embodied_ai_lab_world_model_physics_fidelity.md](../blogs/wechat_embodied_ai_lab_world_model_physics_fidelity.md)
