---
type: entity
tags: [locomotion, rl, humanoid, motion-prior, hmi-opensource-table, repo, roboparty]
status: draft
updated: 2026-07-30
summary: "roboparty_train：以两个Git子模块组织Roboparty训练生态，串起GMR动作准备、AMP与BeyondMimic训练、跑酷任务、ONNX导出和MuJoCo Sim2Sim；适合从工程目录理解数据、算法与部署前验证的分工。"
related:
  - ../tasks/humanoid-locomotion.md
  - ../methods/reinforcement-learning.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/hmi-opensource-projects-coverage.md
sources:
  - ../../sources/repos/roboparty-train.md
  - ../../sources/repos/humanoid-motion-intelligence.md
---

# roboparty_train

[roboparty_train](https://github.com/Roboparty/roboparty_train) 收录于具身智能研究室 [开源项目主表](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md) 的「Locomotion与运动先验」分组，是本库为该入口建立的独立详情节点。

## 一句话定义

以两个Git子模块组织Roboparty训练生态，串起GMR动作准备、AMP与BeyondMimic训练、跑酷任务、ONNX导出和MuJoCo Sim2Sim；适合从工程目录理解数据、算法与部署前验证的分工。

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| RL | Reinforcement Learning | 并行仿真中学习运动策略 |
| AMP | Adversarial Motion Prior | 对抗约束动作接近人类分布 |
| PPO | Proximal Policy Optimization | 腿式/人形 RL 常用算法 |
| Sim2Real | Simulation to Real | 策略从仿真迁移到真机 |

## 为什么重要

- **主表工程定位清晰**：该条目被放在「Locomotion与运动先验」下，说明它服务的是这条人形运动智能问题链上的具体环节，而不是泛泛的链接收藏。
- **可对照开源边界**：主表已概括其可复现范围（训练/推理/部署或仅方法页）；选型时应先读本页「开源状态」，再回官方 README / 项目页核对许可证与平台支持。
- **便于知识库交叉引用**：独立节点让路线图、对比页与 ingest 日志可以稳定链接，避免只在策展列表里「点名」却无法下钻。

## 核心原理

### 在技术路线中的位置

| 字段 | 内容 |
|------|------|
| 主表分组 | Locomotion与运动先验 |
| 官方入口 | https://github.com/Roboparty/roboparty_train |
| 开源状态（据主表） | 已开源（以官方仓库 README 为准） |

主表给出的技术定位可压缩为：

> 以两个Git子模块组织Roboparty训练生态，串起GMR动作准备、AMP与BeyondMimic训练、跑酷任务、ONNX导出和MuJoCo Sim2Sim；适合从工程目录理解数据、算法与部署前验证的分工。

阅读时建议抓住三点：**(1) 输入是什么数据或观测；(2) 输出是参考轨迹、策略、数据还是中间件能力；(3) 公开材料能否支撑训练/部署复现。**

### 流程直觉（对照主表叙事）

```mermaid
flowchart LR
  A["上游数据 / 观测 / 配置"] --> B["roboparty_train"]
  B --> C["下游策略 / 部署 / 评测"]
```

具体模块边界以官方文档为准；本页不替代 README。

## 工程实践

1. **先核入口类型**：若是 GitHub/Gitee 仓库，从 README 的安装、训练与部署章节入手；若是项目页/论文，先确认是否已挂代码或权重。
2. **对齐本体与接口**：人形项目需核对关节顺序、控制频率、观测契约与仿真后端（Isaac / MuJoCo 等）是否与本机栈一致。
3. **按主表定位做消融**：主表强调的可分拆实验切口（例如只换重定向约束、只换部署层）应优先验证，避免一上来全链路重训。
4. **记录开源边界**：若仅有权重、Sim2Sim 或说明文档，不要假设训练管线可复现。

| 检查项 | 建议 |
|--------|------|
| 许可与星标时效 | 以官方仓库页面为准 |
| 支持机器人 / 仿真 | 读 assets 与 task 配置 |
| 真机入口 | 查找 SDK、ROS、ONNX/JIT 导出说明 |

## 局限与风险

- **主表是策展摘要**：细节、指标与许可以一手来源为准；本页只做知识库节点与导航。
- **开源状态可能变化**：标为待发布的项目后续可能放码；已开源仓库也可能拆分或迁移路径。
- **不要与同名论文页混淆**：若本库另有 `paper-*` 深读页，以论文页承载方法细节，本实体页侧重工程入口与选型。

## 关联页面

- [humanoid-locomotion](../tasks/humanoid-locomotion.md)
- [reinforcement-learning](../methods/reinforcement-learning.md)
- [Humanoid Motion Intelligence](./humanoid-motion-intelligence.md)
- [开源主表覆盖索引](../queries/hmi-opensource-projects-coverage.md)

## 参考来源

- [roboparty_train 来源归档](../../sources/repos/roboparty-train.md)
- [Humanoid Motion Intelligence 仓库归档](../../sources/repos/humanoid-motion-intelligence.md)
- [开源项目主表（上游）](https://github.com/RealXiaoze/humanoid-motion-intelligence/blob/main/%E8%AE%BA%E6%96%87%E4%B8%8E%E9%A1%B9%E7%9B%AE/%E5%BC%80%E6%BA%90%E9%A1%B9%E7%9B%AE%E4%B8%BB%E8%A1%A8.md)

## 推荐继续阅读

- [官方入口](https://github.com/Roboparty/roboparty_train)
- [Humanoid Motion Intelligence 知识库实体页](./humanoid-motion-intelligence.md)
