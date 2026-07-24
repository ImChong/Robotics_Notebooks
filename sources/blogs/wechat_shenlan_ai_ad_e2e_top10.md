# 端到端自动驾驶：十大前沿算法盘点

> 来源归档（blog / 微信公众号）

- **标题：** 端到端自动驾驶：十大前沿算法盘点
- **类型：** blog
- **作者：** 深蓝AI / 深蓝学院（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/kb4aNFyCLWMKEVgjiX6F_g
- **关联专辑：** [《自动驾驶核心算法盘点》](https://mp.weixin.qq.com/mp/appmsgalbum?__biz=MzY4NjA5NTgyMQ==&action=getalbum&album_id=4596755873481310212)（模块化栈姊妹篇；本文为端到端续篇）
- **发表日期：** 2026-07-23
- **入库日期：** 2026-07-24
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.5.0 + [wechat-article-for-ai](https://github.com/bzd6661/wechat-article-for-ai)（Camoufox）；短链直连成功；`--no-images`；Jina Reader 对公众号不可用
- **一句话说明：** 按「规划导向 → 向量化 → VLM → 多模态统一 → 世界模型 → 稀疏化 → VLM/E2E 解耦 → 动量一致性 → 并行 Transformer → 截断扩散」线索盘点 UniAD / VAD / DriveVLM / EMMA / GAIA-1 / SparseDrive / Senna / MomAD / DriveTransformer / DiffusionDrive 十项代表性端到端工作。

## 核心摘录（归纳，非全文）

### 问题重框

- 模块化栈（检测→跟踪→预测→规控）可解释、易分工，但存在 **累积误差** 与 **优化目标与安全目标不对齐**。
- 端到端把传感器输入直接映射为驾驶决策，用统一目标学习感知→规划全链路；自 UniAD（CVPR 2023 Best Paper）后快速分化出多条技术主线。

### 十条演进线索与代表工作

| 序 | 线索 | 代表工作 | 核心一句话 |
|----|------|----------|------------|
| 01 | 规划导向联合优化 | **UniAD** | Track/Map/Motion/Occ/Planner 串联；感知服务规划 |
| 02 | 全向量化场景 | **VAD** | 边界/车道/运动/自车向量；弃密集栅格 |
| 03 | VLM 链式思考 | **DriveVLM** | 描述→分析→层次规划；Dual 兼顾实时 |
| 04 | 一切皆语言多模态 | **EMMA** | Gemini 上统一文本化轨迹/检测/建图 |
| 05 | 生成式世界模型 | **GAIA-1** | 视频+文本+动作 token 预测未来帧 |
| 06 | 稀疏中心范式 | **SparseDrive** | 彻底弃稠密 BEV；实例稀疏交互 |
| 07 | VLM 决策 / E2E 数值解耦 | **Senna** | Senna-VLM 高层决策 + Senna-E2E 轨迹 |
| 08 | 帧间轨迹动量 | **MomAD** | 轨迹动量 + 感知动量；量产一致性 |
| 09 | 任务并行 Transformer | **DriveTransformer** | 打破感知→预测→规划级联 |
| 10 | 截断扩散实时规划 | **DiffusionDrive** | 锚点 + 截断去噪；约 2 步 / 45 FPS |

### 收束判断（文内）

1. 感知–规划联合优化仍是上限手段；
2. LLM/VLM 开放世界知识补长尾；
3. 生成式世界模型服务闭环训练与仿真；
4. 可解释性、形式化安全边界与极端泛化仍是量产瓶颈。

## 一手论文索引（文内参考文献）

1. Hu et al., Planning-oriented Autonomous Driving, CVPR 2023（UniAD）— arXiv:2212.10156
2. Jiang et al., VAD: Vectorized Scene Representation…, ICCV 2023 — arXiv:2303.12077
3. Tian et al., DriveVLM…, CoRL 2025 — arXiv:2402.12289
4. Hwang et al., EMMA…, TMLR / arXiv:2410.23262
5. Wayve, GAIA-1…, arXiv:2309.17080
6. Sun et al., SparseDrive…, arXiv:2405.19620
7. Jiang et al., Senna…, arXiv:2410.22313
8. Song et al., Don't Shake the Wheel (MomAD), CVPR 2025 — arXiv:2503.03125
9. Jia et al., DriveTransformer…, ICLR 2025 — arXiv:2503.07656
10. Liao et al., DiffusionDrive…, CVPR 2025 Highlight — arXiv:2411.15139

## 对 wiki 的映射

- 父节点：[e2e-autonomous-driving-top10-algorithms](../../wiki/overview/e2e-autonomous-driving-top10-algorithms.md)
- 姊妹专辑：[autonomous-driving-core-algorithms-series](../../wiki/overview/autonomous-driving-core-algorithms-series.md)
- 十篇论文实体：`wiki/entities/paper-uniad.md` … `paper-diffusiondrive.md`（见 catalog）
- 交叉：[generative-world-models](../../wiki/methods/generative-world-models.md)、[paper-s-squared-vla](../../wiki/entities/paper-s-squared-vla.md)、[paper-m4world](../../wiki/entities/paper-m4world.md)、[vla](../../wiki/methods/vla.md)

## 可信度与使用边界

- 微信「盘点」策展体例；指标与架构细节以 arXiv / 项目页 / 官方仓库为准。
- 原始抓取正文见 [sources/raw/wechat_shenlan_ai_ad_e2e_top10_2026-07-23/](../raw/wechat_shenlan_ai_ad_e2e_top10_2026-07-23/)。
