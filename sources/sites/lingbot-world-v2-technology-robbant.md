# LingBot-World 2.0 / LingBot-World-Infinity（technology.robbyant.com）

> 来源归档（ingest · 项目页）

- **标题：** LingBot-World 2.0 — Infinite Worlds with Versatile Interactions
- **类型：** site / project-page
- **官方入口：** <https://technology.robbyant.com/lingbot-world-v2>
- **论文：** <https://arxiv.org/abs/2607.07534>
- **代码：** <https://github.com/robbyant/lingbot-world-v2>
- **权重集合：** <https://huggingface.co/collections/robbyant/lingbot-world-v2>
- **ModelScope 集合：** <https://modelscope.cn/collections/Robbyant/LingBot-World-V2>
- **机构：** 蚂蚁灵波（Robbyant / Ant Group）
- **入库日期：** 2026-09-10
- **一句话说明：** 720p@60fps、亚秒级延迟的 **可玩** 交互世界模型；Pilot/Director 双 Agent 具身 harness、多人共 steering、长时无视觉漂移；14B + 1.3B 权重与 causal-fast 推理已开源。

## 项目页要点（2026-09-10）

| 能力 | 描述 |
|------|------|
| **Real-Time Infinite Worlds** | 高保真可探索环境；实时推理 + **亚秒级** 控制延迟；720p **60 fps** |
| **Collaborative Steering** | 多用户共享生成世界：Player 导航 + Director 引导事件/高层意图 |
| **Versatile Interaction** | Agent 驱动行为 + 用户指令/事件 grounding；攻击、射箭、施法等多样动作 |
| **Embodied Simulation** | 从 egocentric / 合成 / web 视频学 action-conditioned 视觉动力学，服务机器人仿真与交互数据 |

## 在线体验（非官方推理栈）

- 国际 Web：[Reactor](https://www.reactor.inc/lingbot-world-v2)
- 国内移动端：[LingGuang](https://www.lingguang.com/support)
- 官方完整能力 demo：README 指向 [WAIC 2026](https://waica2026.worldaic.com.cn/)

## 开源核查（步骤 2.5）

| 组件 | 链接 | 结论 |
|------|------|------|
| 推理代码 | [robbyant/lingbot-world-v2](https://github.com/robbyant/lingbot-world-v2) | **已开源**（基于 Wan2.2；`generate.py` + `run_fast.sh`） |
| 权重 | HF 集合 5 项（2026-09-10 全量发布） | **已发布** — 14B causal-fast / causal-pretrain / bid + 1.3B causal-fast |
| 许可 | CC BY-NC-SA 4.0 | 非商业共享；商用需另议 |

## 对 wiki 的映射

- [paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling](../../wiki/entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md)
- [Generative World Models](../../wiki/methods/generative-world-models.md)
- [LingBot-World 1.0（索引页）](../../wiki/entities/paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)

## 当前提炼状态

- [x] 项目页 + GitHub README + HF 集合核查
- [x] 升格既有 arXiv 实体页（不新建重复节点）
