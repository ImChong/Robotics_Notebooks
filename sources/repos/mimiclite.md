# MimicLite

> 来源归档

- **标题：** MimicLite
- **类型：** repo（监督学习 / 运动跟踪基础设施）
- **来源：** Roboparty（GitHub 组织）
- **链接：** https://github.com/Roboparty/MimicLite
- **入库日期：** 2026-07-14
- **一句话说明：** 面向人形机器人通用运动跟踪的开源训练与部署基础设施，贯通数据、训练、统一评测与真机部署，支持小时级迭代与跨 codebase 策略接入。
- **沉淀到 wiki：** 是（`wiki/entities/mimiclite.md`）

---

## 核心能力（文内自述）

| 能力 | 说明 |
|------|------|
| 小时级训练 | 8×RTX 4090、约 3h 训练通用跟踪策略；约 24 GPU-hours |
| Tracking Infra | any4hdmi 统一多来源动作；mjhub 保证训练/运动学/sim2sim 一致性 |
| 遥操 + 高动态 | 同一 policy 支持 Pico/XR 低延迟遥操与高动态真机动作 |
| 跨 codebase 部署 | 模块化 observation interface；已接入 SONIC、HEFT、TeleopIT、Humanoid-GPT、BFM-Zero、TWIST2 |

---

## 关键子组件（文内命名）

- **any4hdmi** — 将 LAFAN、100STYLE、SONIC、真机数据等组织为统一 motion 格式
- **mjhub** — 机器人模型在训练、运动学计算与 sim2sim 验证中的一致性层

---

## 对 wiki 的映射

- [mimiclite](../../wiki/entities/mimiclite.md)
- 父级：[party-os](../../wiki/entities/party-os.md)
- 交叉：[sonic-motion-tracking](../../wiki/methods/sonic-motion-tracking.md)、[motion-retargeting](../../wiki/concepts/motion-retargeting.md)
