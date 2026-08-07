# RekaCS2-10k（新闻发布页）

> 来源归档

- **标题：** RekaCS2-10k: A Large-Scale Egocentric Counter-Strike 2 Dataset
- **类型：** site / news announcement
- **链接：** <https://reka.ai/news/cs2-10k-a-large-scale-egocentric-counter-strike-2-dataset>
- **Hugging Face：** <https://huggingface.co/datasets/RekaAI/CS2-10k>
- **渲染管线：** <https://github.com/reka-ai/cs2-dem-renderer>
- **交互 Viewer：** <https://huggingface.co/spaces/RekaAI/CS2-10k-viewer>
- **机构：** 瑞卡人工智能（Reka AI）
- **发布日期：** 2026-06-24（页面署 Jun 24, 2026）
- **入库日期：** 2026-08-07
- **许可：** 数据集 CC BY-NC 4.0（见 HF card）；渲染器 MIT（见 GitHub）
- **一句话说明：** 从 HLTV 职业 CS2 demo 渲染的 **10,000+ 小时 / 600,000+ 玩家-回合** 第一人称视频，带逐帧键盘、鼠标与 3D 轨迹标注；配套开源 `cs2-dem-renderer`，服务动作条件世界模型训练。

---

## 发布要点

| 维度 | 内容 |
|------|------|
| 规模 | **600,000+** player-round 视频 · **10,000+** 小时第一人称画面 |
| 来源 | 公开职业比赛 demo（[HLTV](https://www.hltv.org/)） |
| 渲染 | CS2 内置 demo replay → **720p · 48 fps**；每玩家每回合一条 |
| 标注 | 同步 parquet：键盘态、鼠标增量、世界坐标、yaw/pitch |
| 视觉处理 | 无中途剪辑、无 HUD；隐藏武器以减少后坐/换弹突变 |
| 相关工作 | [EgoCS-400k](https://egocs-400k.github.io/#dataset) 等 CS ego 数据社区 |

## 下游用例（发布页）

1. **动作条件视频生成** — 当前帧 + 键鼠序列 → 未来 N 帧（GameNGen / Genie / DIAMOND / OASIS 等）
2. **Egocentric 导航先验** — 走廊 vs 开阔点位的前进视觉与相机–位移相关
3. **长程规划** — 回合 60–90 s 量级战术结构（进点 / 守点 / 转点 / 反打）
4. **多智能体世界建模** — 同局 10 名玩家共享 round/map id，可研究动作对他人观测的因果影响

## 渲染管线叙事

`.dem` → 两遍解析（出生/死亡区间 + 逐帧按键）→ 驱动 CS2 demo replay → 电影输出流式送 ffmpeg（VAAPI HEVC）→ `.mp4` + 同步 `.parquet`；worker 模式可批处理整目录并去重。

## 开源状态（项目页核查，2026-08-07）

| 产物 | 状态 | 入口 |
|------|------|------|
| 数据集 | **已开源**（ungated） | [RekaAI/CS2-10k](https://huggingface.co/datasets/RekaAI/CS2-10k) |
| 渲染器 | **已开源**（MIT） | [reka-ai/cs2-dem-renderer](https://github.com/reka-ai/cs2-dem-renderer) |
| Viewer | **已发布** | HF Space + 仓内 `viewer/` |
| 许可边界 | 数据 **CC BY-NC 4.0**（非商用）；底层 demo 版权归原权利人 | HF card |

## 对 wiki 的映射

- **wiki/entities/rekacs2-10k-dataset.md** — 数据集实体页（主升格）
- **sources/datasets/rekacs2-10k.md** — HF 数据卡归档
- **sources/repos/cs2-dem-renderer.md** — 官方渲染器
- **wiki/concepts/world-action-models.md** / **wiki/concepts/video-as-simulation.md** — 动作条件世界模型数据层对照
