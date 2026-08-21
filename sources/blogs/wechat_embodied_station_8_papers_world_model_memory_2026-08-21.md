# 机器人不再只会“看见再行动”：8篇论文揭示世界模型与长期记忆新拐点

> 来源归档（blog / 微信公众号）

- **标题：** 机器人不再只会“看见再行动”：8篇论文揭示世界模型与长期记忆新拐点
- **类型：** blog
- **作者：** 具身智能小站（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/30hu9SRxbRNXJcGLnNwl_g
- **发表日期：** 2026-08-21
- **入库日期：** 2026-08-21
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + `wechat-article-for-ai`（Camoufox）；`--no-images`；Jina Reader 对 `mp.weixin.qq.com` 返回 CAPTCHA，未采用
- **原始抓取落盘：** [`sources/raw/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md`](../raw/wechat_embodied_station_8_papers_world_model_memory_2026-08-21.md)
- **一句话说明：** 汇总 8 篇近期机器人/具身论文，主线从单点策略精度转向「补全隐藏状态—保存长期历史—预测动作后果—约束真实执行」闭环；**8/8 均有独立 `paper-*` 详情节点**（本 ingest **新建 5**、**复用既有 complete 3**；**lint 禁止同一 arXiv 多 canonical 节点，0 重复造页**）。

## 核心摘录（归纳，非全文）

文内判断：当观测不完整、环境持续变化、任务链条变长时，机器人需要同时修复残缺感知、保留跨会话历史、预测行为后果并在线约束不可行命令。8 篇工作分别覆盖双臂抓取局部几何补全、灵巧 RL 技能先验、波动性场景记忆、水下频域增强、令牌级流式 TTS、VR 颗粒喂食仿真、人形行为世界模型与跨本体 action flow。

### 8 篇 → 本库节点

| # | 论文 | arXiv | 开源结论（入库日） | wiki |
|---|------|-------|-------------------|------|
| 01 | PartialBiGrasp | [2608.19188](https://arxiv.org/abs/2608.19188) | **部分开源**：架构仓已建，权重/训练/数据 **TODO** | [paper-partialbigrasp](../../wiki/entities/paper-partialbigrasp.md) |
| 02 | ADEPT | [2608.19182](https://arxiv.org/abs/2608.19182) | **待发布**：项目页 Code → Coming soon | [paper-adept-dexterity](../../wiki/entities/paper-adept-dexterity.md)（既有 complete） |
| 03 | LT-Mem | [2608.19059](https://arxiv.org/abs/2608.19059) | **部分开源**：LT-VQA 数据集可下；代码 **TBD** | [paper-lt-mem](../../wiki/entities/paper-lt-mem.md) |
| 04 | Dynamic SpectraFormer | [2608.18662](https://arxiv.org/abs/2608.18662) | **待发布**：GitHub 仅标题 README | [paper-dynamic-spectraformer](../../wiki/entities/paper-dynamic-spectraformer.md) |
| 05 | X2Streaming-TTS | [2608.18661](https://arxiv.org/abs/2608.18661) | **待发布**：论文链仓库 **404** | [paper-x2streaming-tts](../../wiki/entities/paper-x2streaming-tts.md) |
| 06 | VERAGMIL | [2608.18258](https://arxiv.org/abs/2608.18258) | **待发布**：仓仅 README + GIF | [paper-veragmil](../../wiki/entities/paper-veragmil.md) |
| 07 | GigaBrain-WBC-0.5 | [2608.18234](https://arxiv.org/abs/2608.18234) | **待发布**：项目页 Code → coming soon | [paper-gigabrain-wbc-0-5](../../wiki/entities/paper-gigabrain-wbc-0-5.md)（既有 complete） |
| 08 | Hydra-0 | [2608.18077](https://arxiv.org/abs/2608.18077) | **确认未开源**（项目页无 GitHub/权重） | [paper-hydra-0](../../wiki/entities/paper-hydra-0.md)（既有 complete） |

### 文内要点速记

1. **PartialBiGrasp** — 局部点云 → 占据网络补厚度/边缘 → 力闭合双臂抓取对；不重建完整物体。
2. **ADEPT** — 16 primitive reposing 预训练 + 保守 post-train + visuo-tactile distill；Flexiv 触觉 8/10 vs 视觉 3/10。
3. **LT-Mem** — Live/Delta/Meta 三层记忆 + 波动性更新；LT-VQA 多会话时间问答；令牌消耗低一个数量级。
4. **Dynamic SpectraFormer** — 频域分离低频色偏与高频纹理；动态频谱权重选关键频带。
5. **X2Streaming-TTS** — 令牌级因果 TTS + 语音状态继承；首音频令牌中位时延 15.8 ms。
6. **VERAGMIL** — VR + Isaac Sim 颗粒食物仿真；VR 示范优于 3D 鼠标；BCQ 综合最好。
7. **GigaBrain-WBC-0.5** — 行为世界模型联合预测 action/state/command GMM；地形 SR 81.3%、跌倒恢复 99.3%。
8. **Hydra-0** — action flow 统一跨本体视觉接口；RoboLab 开环 replay 与参考成功率 r=0.96。

## 对 wiki 的映射

- **8/8 独立详情节点**：每篇对应唯一 `wiki/entities/paper-*.md`；静态站 `detail.html?id=entity-paper-…` 均可直达。
- **本 ingest 新建 5** 个实体；**3 篇**在 PR #1642 / 先前 ingest 已有 complete 页 → **仅回链博客与 sources，不新建第二节点**。
- 交叉：[生成式世界模型](../../wiki/methods/generative-world-models.md)、[Sim2Real](../../wiki/concepts/sim2real.md)、[模仿学习](../../wiki/methods/imitation-learning.md)。

## 当前提炼状态

- [x] 公众号正文抓取与 raw 归档
- [x] 8 篇独立节点核查（5 新建 / 3 既有 complete / **0 重复 arXiv 节点**）
- [x] 项目页与仓库开源状态核查（步骤 2.5）
