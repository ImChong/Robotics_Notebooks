# Humanoid Motion Intelligence（人形机器人运动智能知识库）

- **URL**：<https://github.com/RealXiaoze/humanoid-motion-intelligence>（默认分支 `main`）
- **类型**：repo / curated knowledge base（论文解读 + 开源索引 + 数据集 + 产业与求职）
- **维护方**：具身智能研究室（GitHub：`RealXiaoze`）
- **收录日期**：2026-07-28
- **复核日期**：2026-09-10
- **Stars / Forks（本次核查）**：515 ★ / 44 forks（相对 2026-07-28 入库时约 22 ★）
- **最近上游推送**：2026-09-07
- **许可**：分层 — 原创解读 / 技术路线 / 产业与求职编排为 **CC BY-NC-SA 4.0**；公开校验脚本为 **MIT**；论文原图与第三方材料权利归原作者（见仓库 [`LICENSE.md`](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/LICENSE.md)）
- **Tags**：#humanoid #motion-intelligence #curated-list #locomotion #loco-manipulation #vla #sim2real #dataset #career

## 一句话

把人形机器人 **运动智能** 从数据重定向到实机部署串成六条技术路线，并配套 **191 篇论文逐篇解读**、**586 个开源项目主表**、**40 个数据集**、**176 家公司/机构**、按公司拆开的官方开源目录，以及产业信号与求职面经——与本库已 ingest 的「具身智能研究室」微信长文同源，GitHub 主站持续扩容。

## 为什么值得保留

- **与本库微信策展同源**：本仓库是公众号长文（42 篇 RL 栈、64 篇运动小脑、AMP、Loco-Manip、国内开源全景等）的 **GitHub 结构化落地**；适合作为持续更新的外部总入口，而不是再复制一遍论文列表。
- **按问题地图组织，而非按算法标签堆砌**：六条路线（数据 → Locomotion/先验 → 跟踪/WBC → LocoManip → WM/VLA/Agent → 工程部署）与本库 [身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) / [运动小脑地图](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md) **视角可对照**。
- **开源状态写进索引**：论文表与开源主表对「是 / 部分 / 否 / 待发布」标注较细，利于选型与复现判断。
- **产业、数据集与求职侧栏**：公司主表、公开信号时间线、数据集主表（`Dxxx`）、面经与招聘快照对本库主线是 **旁路入口**（使用前必须回原始招聘页 / 数据卡核对时效）。

## 开源核查（2026-09-10）

| 项 | 结论 |
|---|---|
| 仓库可见性 | **已公开**（GitHub `main`；515 ★ / 44 forks） |
| 项目页 | 无独立 `*.github.io` 项目页；以仓库 README 为导航入口 |
| 内容形态 | Markdown 知识库 + `.github` 公开树校验工作流（`public-release.json` 约 503 个内容文件）；**不是**可训练的算法实现仓 |
| 可运行训练/推理入口 | **不适用**（无 `train.py` / 策略权重；复现入口指向各论文官方仓） |
| 许可 | 原创编排 **CC BY-NC-SA 4.0**；校验脚本 **MIT**；第三方论文图 / 代码许可证不变 |

## 仓库结构（维护者视角）

| 目录 | 作用（本次规模） |
|------|------------------|
| [`技术路线/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF) | 六条路线总览 + 新手七阶段学习路径（含系统能力栈 / 训练闭环） |
| [`论文与项目/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) | 论文总索引 **191** 条（`P001`–`P191`）+ `论文逐篇解读/` + [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) **586** 项 |
| [`数据集/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E6%95%B0%E6%8D%AE%E9%9B%86) | **新增目录**：40 个数据集主表（稳定 `Dxxx`）+ Open-AoE / RealOmni-Open 等数据链路专页 |
| [`具身智能公司的开源项目/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E5%85%B7%E8%BA%AB%E6%99%BA%E8%83%BD%E5%85%AC%E5%8F%B8%E7%9A%84%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE) | **新增目录**：按公司拆页的官方开源（约 81 家页面；只收可核验归属的公开仓） |
| [`强化学习开发者必备开源资料/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E5%BC%BA%E5%8C%96%E5%AD%A6%E4%B9%A0%E5%BC%80%E5%8F%91%E8%80%85%E5%BF%85%E5%A4%87%E5%BC%80%E6%BA%90%E8%B5%84%E6%96%99) | **新增目录**：Gymnasium / CleanRL / SB3 / Isaac Lab + RSL-RL 分层选型，而非堆仓库 |
| [`公司与产业/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E5%85%AC%E5%8F%B8%E4%B8%8E%E4%BA%A7%E4%B8%9A) | 公司主表 **176** 家、公开信号时间线；明确不做强弱排名 |
| [`求职与岗位/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E6%B1%82%E8%81%8C%E4%B8%8E%E5%B2%97%E4%BD%8D) | 运控面经、秋招问答、招聘/内推快照（时效敏感） |
| `.github/` | `public-release.json` + 树校验脚本 / CI，约束公开树与清单一致性 |
| [`AGENTS.md`](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/AGENTS.md) | 上游给 Agent 的检索规则：先读技术路线，再用 `Pxxx` / 项目名定位，区分论文结论与 README 声明 |

### 六条技术路线（README 当前计数）

| 路线 | 论文 / 项目（上游自报） | 本库邻近入口 |
|------|------------------------|--------------|
| 动作数据与重定向 | 17 篇 / 47 项 | [GMR](../../wiki/methods/motion-retargeting-gmr.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md) |
| Locomotion 与运动先验 | 39 篇 / 61 项 | [Locomotion](../../wiki/tasks/locomotion.md)、[AMP 综述](../../wiki/overview/humanoid-amp-motion-prior-survey.md) |
| 动作跟踪与全身控制 | 40 篇 / 52 项 | [SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md) |
| LocoManip | 31 篇 / 37 项 | [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) |
| 世界模型、VLA 与 Agent | 47 篇 / 103 项 | [VLA](../../wiki/methods/vla.md)、[世界模型 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) |
| 工程与实机部署 | 17 篇 / 286 项 | [Sim2Real](../../wiki/concepts/sim2real.md)、[运动控制主路线](../../roadmap/motion-control.md) |

项目数在工程层暴涨（286），是因为本体接口、仿真器、控制库、数据工具和部署中间件都汇到这一层；**不要把条目数当成方法贡献**。

## 与本库已有资料的关系

- **不要整仓镜像进 wiki**：本库继续按实体/方法/任务编译；本仓作 **外部策展总入口**。
- **论文导读快照**：2026-07-31 已覆盖当时的 P001–P145；2026-09-10 复核把 P146–P191 接到已有详情页（少数缺口见 [HMI 论文导读](../../wiki/queries/hmi-papers-coverage.md)）。
- **开源主表快照**：2026-07-30 导读覆盖当时的 166 项；上游现为 **586** 项，本库 **不镜像全表**，读者以主表原文为准。
- **微信姊妹篇已入库**（示例）：
  - [42 篇 RL 运动控制](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
  - [运动小脑 64 篇](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
  - [AMP 运动先验](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
  - [国内开源全景 76 家 424 项](../blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- **互补清单**：[awesome-humanoid-robot-learning](awesome-humanoid-robot-learning.md)（Yanjie Ze，偏真机+开源论文列表）；[Robot Learning Paper Notebooks](https://github.com/ImChong/Robot_Learning_Paper_Notebooks)（单篇深读）；飞书 [开源运动控制项目](../../wiki/queries/open-source-motion-control-projects.md)（上游 README 另链「小而美的运动控制项目」）。

## 对 wiki 的映射

- 升格实体页：[humanoid-motion-intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
- 论文导读：[hmi-papers-coverage](../../wiki/queries/hmi-papers-coverage.md)
- 开源主表导读：[hmi-opensource-projects-coverage](../../wiki/queries/hmi-opensource-projects-coverage.md)
- 交叉：身体系统栈、运动小脑地图、开源运动控制项目 query、运动控制主路线

## 使用边界

- 论文结论、开源状态、公司与招聘信息 **以原始 arXiv / 官方仓 / 招聘页 / 数据卡为准**；本仓为第三方策展。
- 「有代码 / 有演示」不等于完整可复现或稳定真机部署（上游 `AGENTS.md` 同样强调）。
- CC BY-NC-SA 约束原创解读的转载方式；上游代码许可证不因本库文字许可而改变。
