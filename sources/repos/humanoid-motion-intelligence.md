# Humanoid Motion Intelligence（人形机器人运动智能知识库）

- **URL**：<https://github.com/RealXiaoze/humanoid-motion-intelligence>（默认分支 `main`）
- **类型**：repo / curated knowledge base（论文解读 + 开源索引 + 产业与求职）
- **维护方**：具身智能研究室（GitHub：`RealXiaoze`）
- **收录日期**：2026-07-28
- **Stars（入库核查）**：约 22
- **许可**：分层 — 原创解读 / 技术路线 / 产业与求职编排为 **CC BY-NC-SA 4.0**；公开校验脚本为 **MIT**；论文原图与第三方材料权利归原作者（见仓库 [`LICENSE.md`](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/LICENSE.md)）
- **Tags**：#humanoid #motion-intelligence #curated-list #locomotion #loco-manipulation #vla #sim2real #career

## 一句话

把人形机器人 **运动智能** 从数据重定向到实机部署串成六条技术路线，并配套 **~145 篇论文逐篇解读**、**~166 个开源项目主表**、公司产业信号与求职面经——与本库已 ingest 的「具身智能研究室」微信长文同源、结构更完整。

## 为什么值得保留

- **与本库微信策展同源**：本仓库是公众号长文（42 篇 RL 栈、64 篇运动小脑、AMP、Loco-Manip 等）的 **GitHub 结构化落地**；适合作为持续更新的外部总入口，而不是再复制一遍论文列表。
- **按问题地图组织，而非按算法标签堆砌**：六条路线（数据 → Locomotion/先验 → 跟踪/WBC → LocoManip → WM/VLA/Agent → 工程部署）与本库 [身体系统栈](../../wiki/overview/humanoid-rl-motion-control-body-system-stack.md) / [运动小脑地图](../../wiki/overview/humanoid-motion-cerebellum-technology-map.md) **视角可对照**。
- **开源状态写进索引**：论文表与开源主表对「是 / 部分 / 否 / 待发布」标注较细，利于选型与复现判断。
- **产业与求职侧栏**：公司主表、公开信号时间线、面经与秋招问答对本库主线是 **旁路入口**（使用前必须回原始招聘页核对时效）。

## 开源核查（2026-07-28）

| 项 | 结论 |
|---|---|
| 仓库可见性 | **已公开**（GitHub `main`） |
| 内容形态 | Markdown 知识库 + `.github` 公开树校验工作流；**不是**可训练的算法实现仓 |
| 可运行训练/推理入口 | **不适用**（无 `train.py` / 策略权重；复现入口指向各论文官方仓） |
| 项目页 | 无独立 `*.github.io` 项目页；以仓库 README 为导航入口 |

## 仓库结构（维护者视角）

| 目录 | 作用 |
|------|------|
| [`技术路线/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E6%8A%80%E6%9C%AF%E8%B7%AF%E7%BA%BF) | 六条路线总览 + 新手学习路径（含 Mermaid 闭环图） |
| [`论文与项目/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE) | 论文总索引（当前约 **145** 条）+ `论文逐篇解读/Pxxx.md` + [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)（约 **166** 项） |
| [`公司与产业/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E5%85%AC%E5%8F%B8%E4%B8%8E%E4%BA%A7%E4%B8%9A) | 公司主表、公开信号时间线、六类专题；明确不做强弱排名 |
| [`求职与岗位/`](https://github.com/RealXiaoze/humanoid-motion-intelligence/tree/main/%E6%B1%82%E8%81%8C%E4%B8%8E%E5%B2%97%E4%BD%8D) | 运控面经、秋招问答、招聘/内推快照（时效敏感） |
| `.github/` | `public-release.json` + 树校验脚本 / CI，约束公开树与清单一致性 |

### 六条技术路线（与本库对照）

| 路线 | 核心问题 | 本库邻近入口 |
|------|----------|--------------|
| 动作数据与重定向 | 人体/视频/MoCap → 机器人可训练参考 | [GMR](../../wiki/methods/motion-retargeting-gmr.md)、[Motion Retargeting](../../wiki/concepts/motion-retargeting.md) |
| Locomotion 与运动先验 | 基础移动、地形、AMP/行为先验 | [Locomotion](../../wiki/tasks/locomotion.md)、[AMP 综述](../../wiki/overview/humanoid-amp-motion-prior-survey.md) |
| 动作跟踪与全身控制 | 参考跟踪、失配恢复、人体驱动 | [SONIC](../../wiki/methods/sonic-motion-tracking.md)、[BeyondMimic](../../wiki/methods/beyondmimic.md) |
| LocoManip | 移动 + 接触 + 操作改变场景状态 | [Loco-Manipulation](../../wiki/tasks/loco-manipulation.md) |
| 世界模型、VLA 与 Agent | 预测、生成、技能调度与长时任务 | [VLA](../../wiki/methods/vla.md)、[世界模型 taxonomy](../../wiki/overview/robot-world-models-training-loop-taxonomy.md) |
| 工程与实机部署 | 本体接口、Sim2Real、安全评测 | [Sim2Real](../../wiki/concepts/sim2real.md)、[运动控制主路线](../../roadmap/motion-control.md) |

## 与本库已有资料的关系

- **不要整仓镜像进 wiki**：本库继续按实体/方法/任务编译；本仓作 **外部策展总入口**。
- **微信姊妹篇已入库**（示例）：
  - [42 篇 RL 运动控制](../blogs/wechat_embodied_ai_lab_humanoid_rl_motion_survey.md)
  - [运动小脑 64 篇](../blogs/wechat_embodied_ai_lab_humanoid_motion_cerebellum_survey.md)
  - [AMP 运动先验](../blogs/wechat_embodied_ai_lab_humanoid_amp_motion_prior_survey.md)
- **互补清单**：[awesome-humanoid-robot-learning](awesome-humanoid-robot-learning.md)（Yanjie Ze，偏真机+开源论文列表）；[Robot Learning Paper Notebooks](https://github.com/ImChong/Robot_Learning_Paper_Notebooks)（单篇深读）。

## 对 wiki 的映射

- 升格实体页：[humanoid-motion-intelligence](../../wiki/entities/humanoid-motion-intelligence.md)
- 交叉：身体系统栈、运动小脑地图、开源运动控制项目 query、运动控制主路线

## 使用边界

- 论文结论、开源状态、公司与招聘信息 **以原始 arXiv / 官方仓 / 招聘页为准**；本仓为第三方策展。
- CC BY-NC-SA 约束原创解读的转载方式；上游代码许可证不因本库文字许可而改变。
