# 踩遍 VLN 开源项目的坑后，最推荐新手复现的是这 4 个……（亲测复现率 100%）

> 来源归档（blog / 微信公众号）

- **标题：** 踩遍VLN开源项目的坑后，最推荐新手复现的是这4个……（亲测复现率100%）
- **类型：** blog
- **作者：** 深蓝具身智能（微信公众号）
- **原始链接：** https://mp.weixin.qq.com/s/AzCDukzwrfIyms_65kh1mg
- **发表日期：** 2026-01-24（frontmatter）
- **入库日期：** 2026-05-22
- **抓取方式：** [Agent Reach](https://github.com/Panniantong/Agent-Reach) v1.4.0 + `wechat-article-for-ai`（Camoufox）；正文约 1.8 万字 / 22 图
- **姊妹篇：** [VLN 发展历程中 4 个代表性项目复现](https://mp.weixin.qq.com/s?__biz=MzkwMDcyNDUzMQ==&mid=2247496491&idx=1&sn=9ed38a5f612d0e3b670f3d6a74a8d5d7)（文内推荐区外链，未单独 ingest）
- **一句话说明：** 按 VLN 四条范式各选 1 个「能跑通」开源栈：模块化语义地图（VLFM）、纯 LLM 推理（NavGPT）、扩散统一探索/到达（NoMaD）、统一 VLA 序列预测（Uni-NaVid）；强调由浅入深学习路径而非性能榜单。

## 核心摘录（归纳，非全文）

### 选题立场

- 目标读者：**初学者**；筛选标准 = 可上手、可运行、可理解（作者称亲测复现率 100%）。
- 不追求 SOTA 对比；关注 `python run.py` 级可复现性。

### 四范式 × 四项目

| 范式 | 项目 | 机构/备注 | GitHub | 复现要点（归纳） |
|------|------|-----------|--------|------------------|
| **2D 语义代价地图 / 模块化** | **VLFM** | Boston Dynamics AI Institute 等 | `bdaiinstitute/vlfm` | Habitat + HM3D；深度图→几何地图 + VLM 语义→frontier/value map；无需训练；CUDA 推理即可 |
| **显式 LLM 语言推理** | **NavGPT** | — | `GengzeZhou/NavGPT` | R2R；视觉转文本描述 + prompt 推理动作；**OpenAI API**；几乎无本地 GPU 训练 |
| **端到端扩散策略** | **NoMaD** | ICRA 2024 最佳论文相关叙事 | `adith-m-dharan/NoMaD` | 连续观测→扩散采样动作轨迹；**goal masking** 统一探索与目标导航；预训练权重 + 可选 ROS/LoCoBot |
| **统一 VLA 序列预测** | **Uni-NaVid** | RSS 2025 | `jzhzhang/Uni-NaVid` | 在线 RGB + 指令→动作 token 序列；多任务统一；**算力高**（文内 A100 ~5Hz 推理） |

### 技术演进线（作者收束）

1. **VLFM** — 地图 + 前沿 + 语义价值（经典机器人导航直觉）
2. **NavGPT** — 决策中枢 = LLM prompt（零样本探索）
3. **NoMaD** — 无显式地图的 e2e 扩散策略
4. **Uni-NaVid** — 导航即 VLA 跨模态序列生成

### 各项目复现路径摘要

**VLFM：** Conda + 可编辑安装 → HM3D + MobileSAM/GroundingDINO/YOLOv7 权重 → 启动 VLM 服务 → Habitat 评测。核心代码目录 `vlfm/vlm`（`reset`/`step`/`_get_obs`）。

**NavGPT：** Python 3.9 → R2R 数据 → API Key → `NavAgent.test()` 验证集推理；可用 GPT-3.5 小样本 эконом。

**NoMaD：** 优先加载官方预训练 → 理解 goal masking 行为差 → 再考虑 `navigate.sh`/`explore.sh` 与 ROS；扩散 denoise 循环 + waypoint 发布。

**Uni-NaVid：** Conda（Py3.10、flash-attn）→ 下载预训练 VLA 权重 → 小规模示例/离线评估；`UniNaVid_Agent.act()` 解析 forward/left/right/stop。

### 与 VLA 综述的区别

- 本站已 ingest 的 [VLA GitHub 复现景观](../../wiki/overview/vla-open-source-repro-landscape-2025.md) 偏 **操作/manipulation VLA**；本篇专 **VLN 导航**。
- **Uni-NaVid**（导航 VLA）≠ **UniVLA**（跨平台潜动作，见 VLA 景观页）。

## 对 wiki 的映射

- [vln-open-source-repro-paradigms](../../wiki/overview/vln-open-source-repro-paradigms.md)（本次升格主页面）
- [vision-language-navigation](../../wiki/tasks/vision-language-navigation.md)、[vla](../../wiki/methods/vla.md)、[diffusion-policy](../../wiki/methods/diffusion-policy.md)

## 可信度与使用边界

- 「复现率 100%」为作者亲测表述，环境与依赖版本因人而异。
- 推广课程/直播信息已剥离；API 费用与 HM3D 数据许可需自行确认。

## 当前提炼状态

- [x] 抓取与四项目索引
- [x] wiki 总览页与任务页交叉链接
