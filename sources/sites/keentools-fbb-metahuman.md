# FaceBuilder × MetaHuman（KeenTools 集成页）

- **类型**：网站 / 集成工作流说明
- **入口**：<https://keentools.io/integrations/fbb-mh>
- **主体**：KeenTools（与 Epic MetaHuman 生态对接）
- **收录日期**：2026-09-24
- **抓取说明**：以 **2026-09-24** 对集成页公开文案的抓取为准；MetaHuman 插件与 UE 版本以 Epic 文档为准。
- **关联产品页**：[keentools-facebuilder-blender.md](./keentools-facebuilder-blender.md)
- **MetaHuman 文档**：[metahuman-epic-docs.md](./metahuman-epic-docs.md)

## 一句话

**FaceBuilder for Blender + Mesh to MetaHuman** 官方五步法：用 **4–8 张照片** 在 Blender 内建 **写实 3D 头**，导出 **MetaHuman 兼容 MH texture**，经 **Mesh to MetaHuman** 对齐拓扑后在 **MetaHuman Creator** 定制，并把 FaceBuilder 纹理（含痣、疤痕等细节）应用到最终 MetaHuman。

## 为什么值得保留

- 这是 **照片 → Blender 头部 → UE MetaHuman** 的 **官方 documented 捷径**，与机器人栈中 **人类化身、演示级数字孪生、表演可视化** 直接相关。
- 明确 **MH UV / MH texture** 版本门槛（2022.2+），便于 wiki 与 [MetaHuman](../../wiki/entities/metahuman.md) 实体页交叉引用 **Mesh to MetaHuman** 小节。
- 与 Epic 文档中的 **Mesh to MetaHuman（全身）** 互补：本集成页侧重 **FaceBuilder 驱动的面部身份与皮肤纹理** 迁移。

## 五步工作流（页内标题「Yourself to MetaHuman in 5 Steps」）

| 步骤 | 动作 |
|------|------|
| 1 | 拍摄 **4–8 张** 照片，载入 FaceBuilder for Blender |
| 2 | 构建 **3D 头**；生成 **MH texture**（与 MetaHuman 拓扑/UV 兼容） |
| 3 | 在 **Mesh to MetaHuman** 中将 3D 头对齐 MetaHuman 拓扑 |
| 4 | 在 **MetaHuman Creator** 中定制模型（发型、服装等） |
| 5 | 应用 FaceBuilder 导出的 **MH texture** |

```mermaid
flowchart LR
  PHOTOS[4-8 张照片] --> FB[FaceBuilder for Blender]
  FB --> HEAD[3D 头 + MH texture]
  HEAD --> M2MH[Mesh to MetaHuman]
  M2MH --> CREATOR[MetaHuman Creator]
  CREATOR --> APPLY[应用 MH texture]
  APPLY --> MH[可动画 MetaHuman]
```

## 系统与许可（页内 FAQ 摘要）

| 主题 | 说明 |
|------|------|
| **FaceBuilder 环境** | 官方 64 位 Blender 2.80+、可用 3D 视口的 GPU；Win/Linux/macOS 64 位 |
| **MetaHuman 环境** | 指向 Epic MetaHuman 系统要求页 |
| **FaceBuilder 许可** | 付费插件，**15 天试用** |
| **MetaHuman 插件许可** | 以 Epic 许可协议为准 |
| **兼容版本** | FaceBuilder **2022.2 起** 支持 **MH UV mapping** |

## 对 wiki 的映射

- 主实体页：[wiki/entities/keentools-facebuilder.md](../../wiki/entities/keentools-facebuilder.md)（集成专节）
- MetaHuman 实体：[wiki/entities/metahuman.md](../../wiki/entities/metahuman.md)
- Blender 实体：[wiki/entities/blender.md](../../wiki/entities/blender.md)

## 参考链接

- 集成页：<https://keentools.io/integrations/fbb-mh>
- FaceBuilder 产品页：<https://keentools.io/products/facebuilder-for-blender>
- MetaHuman 文档 — Mesh to MetaHuman：<https://dev.epicgames.com/documentation/metahuman/metahuman-creator>
