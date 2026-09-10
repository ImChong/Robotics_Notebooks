# Infinite Worlds with Versatile Interactions (LingBot-World 2.0)

> 来源归档（ingest）

- **标题：** Infinite Worlds with Versatile Interactions
- **别名：** LingBot-World 2.0 / **LingBot-World-Infinity**
- **类型：** paper / interactive-world-model / video-generation / embodied-simulation
- **arXiv：** <https://arxiv.org/abs/2607.07534>（PDF: <https://arxiv.org/pdf/2607.07534>）
- **项目页：** <https://technology.robbyant.com/lingbot-world-v2>
- **代码：** <https://github.com/robbyant/lingbot-world-v2>
- **权重：** <https://huggingface.co/collections/robbyant/lingbot-world-v2>
- **机构：** 蚂蚁灵波（Robbyant / Ant Group）
- **入库日期：** 2026-09-10
- **一句话说明：** LingBot-World 2.0 在 1.0 视频世界模型上引入 **无界交互视界**、**720p@60fps 实时 causal-fast 蒸馏**、**多样交互元素** 与 **Pilot/Director 双 Agent harness**；14B 主模型 + 1.3B 单卡部署；代码与全量变体权重已开源（CC BY-NC-SA 4.0）。

## 摘录 1：四大升级（摘要）

1. **Unbounded Interaction Horizon** — 因果预训练范式，长时交互质量一致、视觉漂移受控。
2. **Rapid Response** — 从基座蒸馏 **causal-fast** 变体，支撑 **720p 60 fps** 与 **亚秒级** 控制延迟。
3. **Diverse Interactive Elements** — 攻击、射箭、施法、射击等动作 + 更丰富 **文本驱动事件**。
4. **Agentic Harness** — **Pilot** 规划/执行角色行为；**Director** 合成随场景推进的新环境元素；支持 **多人共 steering**。

**对 wiki 的映射：** 升格 [`wiki/entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md`](../../wiki/entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md)。

## 摘录 2：Embodied Simulation 叙事

- 项目页：从 **egocentric、合成与 web 规模视频** 学习 **action-conditioned 视觉动力学**。
- 定位：超越游戏式可玩世界，作为 **机器人仿真、未来状态预测、交互数据生成** 的基础。

## 摘录 3：模型与推理（README）

| 变体 | 用途 |
|------|------|
| **14B causal-fast** | 默认实时交互；4 steps/chunk，无 CFG |
| **14B causal-pretrain** | 预训练因果基座；40 steps/chunk + CFG |
| **14B bid** | 双向模型 |
| **1.3B causal-fast** | 单 GPU 轻量部署 |

- **推理：** `generate.py` — chunk-wise causal + **KV caching**（`local_attn_size` / `sink_size`）；输入 **初始帧 + action_path + prompt**。
- **基座代码栈：** Wan2.2（`wan` 包、`i2v-A14B` task）。

## 摘录 4：开源边界

| 项 | 结论 |
|----|------|
| 代码 | **已开源** — GitHub |
| 权重 | **已发布** — HF + ModelScope（2026-09-10 补全 14B pretrain/bid + 1.3B fast） |
| 许可 | **CC BY-NC-SA 4.0** |
| 在线 demo | Reactor / LingGuang 为第三方托管体验；与官方 WAIC demo 能力可能有差 |

## 对 wiki 的映射（汇总）

- [paper-sa-2607-07534 实体页](../../wiki/entities/paper-sa-2607-07534-infinite-worlds-with-versatile-interactions-ling.md)
- [LingBot-World 1.0 索引](../../wiki/entities/paper-sa-2601-20540-advancing-open-source-world-models-lingbot-world.md)
- [Generative World Models](../../wiki/methods/generative-world-models.md)
- [Awesome WM 策展摘录](./sun_awesome_wm_2607_07534_infinite-worlds-with-versatile-interacti.md)

## 当前提炼状态

- [x] arXiv 摘要 + 项目页 + README 核对
- [x] 步骤 2.5：代码与权重已开源
- [x] 复用既有 arXiv 实体页，不新建重复节点
