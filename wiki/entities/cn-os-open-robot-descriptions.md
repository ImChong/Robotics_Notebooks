---
type: entity
tags: [repo, china-embodied-opensource, open-source, project]
status: draft
updated: 2026-09-06
related:
  - ../overview/china-domestic-embodied-opensource-76-companies-technology-map.md
  - ../entities/humanoid-motion-intelligence.md
  - ../queries/china-domestic-opensource-424-coverage.md
sources:
  - ../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md
  - ../../sources/repos/open-robot-descriptions.md
summary: "玄雅科技 开源项目 Open-Robot-Descriptions（本体模型资产）：开源机器人模型描述文件集合…"
institutions:
  - xuanya
---

# Open-Robot-Descriptions

## 一句话定义

**Open-Robot-Descriptions** 是 [玄雅科技](https://github.com/Synria-Robotics) 公开的 **本体模型资产** 开源项目：开源机器人模型描述文件集合

## 英文缩写速查

| 缩写 | 英文全称 | 简要说明 |
|------|----------|----------|
| SDK | Software Development Kit | 真机控制与状态读取接口 |
| RL | Reinforcement Learning | 强化学习训练与策略优化 |
| VLA | Vision-Language-Action | 视觉–语言–动作统一策略 |
| Sim2Real | Simulation to Real | 仿真策略迁移真机 |
| URDF | Unified Robot Description Format | 机器人描述与仿真资产 |

## 为什么重要

- 收录于 [国内具身智能开源全景（76 家 · 424 项）](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md) 的 **第五层** 分组。
- 与 [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md) 同源策展；本页为 **独立详情节点**，便于从公司清单跳到机制与入口说明。

## 核心原理

| 字段 | 内容 |
|------|------|
| 机构 | 玄雅科技 |
| 类别 | 本体模型资产 |
| 官方组织 | https://github.com/Synria-Robotics |

## 工程实践

1. 从官方 GitHub/Gitee 组织检索 `Open-Robot-Descriptions` 仓库并核对 README 许可与依赖。
2. 对照本库 [424 项覆盖索引](../queries/china-domestic-opensource-424-coverage.md) 查看同公司其它入口是否共用训练/部署链路。
3. 若与既有方法页（如 RL 框架、VLA、SDK）主题相同，优先读关联页中的「开源入口」小节，避免重复维护平行叙事。

## 局限与风险

- 公众号清单为 **策展快照**（2026-09-06）；仓库更名、归档或许可证变化须回官方组织页核实。
- **开源状态**：以仓库 README 与 release 为准（入库日按文章描述归纳，未逐仓 clone 验证）。

## 源码运行时序图

仓库主体是 URDF / MJCF / mesh 资产；可运行部分仅是一个 Python 路径查找包 `openrd`。入口：`python examples/01_get_path.py --name unitree_g1 [--version ...] [--variant ...] [--model_format urdf|mjcf]`。

```mermaid
sequenceDiagram
    participant EX as examples/01_get_path.py
    participant OR as openrd.get_model_path（openrd/__init__.py）
    participant FM as openrd.urdf / openrd.mjcf（子包）
    participant RB as 机器人子包（如 openrd/urdf/unitree_g1/__init__.py）
    participant LM as list_available_models
    EX->>OR: get_model_path(name, version, variant, model_format)
    OR->>OR: model_format 选择 urdf 或 mjcf 子包；有 version 时模块名为 name_version
    OR->>FM: getattr(model_module, version_module_name)
    alt 模块不存在
        OR->>LM: list_available_models(show_path=True)
        OR-->>EX: ValueError（附可用模型列表）
    else 找到机器人子包
        FM-->>OR: RB（内含 SimpleNamespace，属性 .urdf / .xml 为用 _MODULE_PATH 拼出的文件路径）
        OR->>RB: 有 variant 则 getattr(RB, variant)；否则遍历属性找带 .urdf（mjcf 为 .xml）的对象
        RB-->>OR: 绝对路径
        OR-->>EX: 返回模型文件路径并打印
    end
```

图依据：Synria-Robotics/Open-Robot-Descriptions@f3e0b6a，读过 README.md（开头）、openrd/__init__.py（get_model_path 大部分）、openrd/urdf/__init__.py、openrd/urdf/unitree_g1/__init__.py、examples/01_get_path.py、auto_generate_init.py（开头）；get_model_path 末尾返回部分与 list_available_models 实现未读。

## 关联页面

- [国内具身开源全景技术地图](../overview/china-domestic-embodied-opensource-76-companies-technology-map.md)
- [HMI 开源项目主表导读](../queries/hmi-opensource-projects-coverage.md)
- [Humanoid Motion Intelligence](../entities/humanoid-motion-intelligence.md)

## 参考来源

- [Open-Robot-Descriptions 源码归档](../../sources/repos/open-robot-descriptions.md)（<https://github.com/Synria-Robotics/Open-Robot-Descriptions>）

- [国内具身智能开源全景（微信公众号）](../../sources/blogs/wechat_embodied_station_domestic_opensource_panorama_2026-09-06.md)
- 源码核对（2026-11-02）：Synria-Robotics/Open-Robot-Descriptions@f3e0b6a，读过 README.md（开头）、openrd/__init__.py（get_model_path 大部分）、openrd/urdf/__init__.py、openrd/urdf/unitree_g1/__init__.py、examples/01_get_path.py；仓库主体为模型资产，仅路径查找包可运行；时序图依据以上文件，未运行代码。

## 推荐继续阅读

- [玄雅科技 官方组织](https://github.com/Synria-Robotics)
