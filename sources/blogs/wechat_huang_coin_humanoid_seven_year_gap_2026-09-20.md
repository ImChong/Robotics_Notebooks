# wechat_huang_coin_humanoid_seven_year_gap_2026-09-20

> 来源归档（blog / 微信公众号）

- **标题：** 人形机器人距离「像个孩子一样聪明」还缺什么？
- **类型：** blog
- **作者：** 黄先生coin
- **原始链接：** https://mp.weixin.qq.com/s/_pW7Pe8-3BW7PS7_pLvccg
- **入库日期：** 2026-09-20
- **抓取方式：** WebFetch（`mp.weixin.qq.com`；本环境未预装 `wechat-article-for-ai`）
- **原始抓取落盘：** [`sources/raw/wechat_huang_coin_humanoid_seven_year_gap_2026-09-20.md`](../raw/wechat_huang_coin_humanoid_seven_year_gap_2026-09-20.md)
- **一句话说明：** 用「一岁会抓杯、七岁才过马路」比喻 VLA 已会抓与仍缺认知可靠性之间的鸿沟；归纳达到「七岁状态」的八种能力、五层缺口、compound reliability 数学与长时记忆+世界模型+可验证安全三支柱。
- **步骤 2.5（开源核查）：** 科普文引用 Gemini Robotics 2、Figure Helix、Isaac GR00T N1.7、π0.7、PI 长短时记忆等公开产品/论文叙事；无单一待核查项目页，以官方仓库与博客为准。
- **沉淀到 wiki：** [`wiki/concepts/humanoid-cognitive-reliability-gap.md`](../../wiki/concepts/humanoid-cognitive-reliability-gap.md)

## 核心摘录（归纳，非全文）

### 核心比喻

| 能力 | 类比 | 本质 |
|------|------|------|
| **抓杯子** | ~1 岁 | 手眼协调 + 目标锁定 → **动作问题** |
| **过马路** | ~7 岁（有限监督） | 感知、规则、他人、风险、抑制冲动 → **认知系统** |

关键洞见：**世界不会因为我学会了规则，就自动按规则运行** — 司机分心、遮挡、他人误判都要纳入。

### VLA 进展（文内样例）

- **Gemini Robotics 2 / ER 2 / On-Device 2** — 全身、灵巧、多机协作；仍偏厂商展示
- **Figure Helix** — 慢 VLM（~7–9 Hz）+ 快控制（~200 Hz）；双机协作、陌生物抓取
- **Isaac GR00T N1.7** — 开放通用 VLA、跨形态迁移、人类视频
- **π0.7** — 跨机器人/场景/任务的技能组合泛化

趋势：从「动作库」→「技能模型」；距「七岁状态」仍远。

### 情境常识缺口

识别杯子 ≠ 理解：是否热水、是否他人物品、会否打翻、旁边是否有幼儿。同一动作在不同场景 **风险完全不同**。

### 「七岁状态」八种能力

1. **持续感知** — 稳定 3D 模型与追踪，非短视频快照
2. **部分可观测 / 物体恒常性** — 遮挡后仍记住位置与任务进展（PI 2026 记忆 ~15 min 为起点）
3. **长期记忆** — 环境习惯、人物偏好、失败经验提炼（非录像堆叠）
4. **因果 / 反事实推理** — 行动前比较「现在做 vs 等十秒」等后果
5. **失败恢复** — 重试、换策略、求助，非一次成功/彻底失败
6. **风险与不确定性表达** — 「看不清」「请确认药品」「前方有人暂停」
7. **社会意图与边界** — 「收拾一下」≠ 乱塞；「快一点」≠ 撞人
8. **稳定身体与可靠硬件** — 触觉、柔顺、足底、抗摔；有人/家具/宠物环境的安全工程

### 五层缺口

1. 真实世界机器人数据（昂贵、长尾）
2. 长时任务数据（分钟级实验 vs 小时/天级家务）
3. 语言→动作中间表示与完成标准
4. 安全评估（误抓/碰撞/误操作/恢复/最坏情况，非仅成功率）
5. 硬件–软件–责任边界（日志、权限、急停）

### Compound reliability

100 步任务、每步 99% → 整体 **~36.6%**；要整体 90% 需每步 **~99.9%**。家庭任务还有变房间、变物体、变成员。

**竞争焦点：** 演示成功 → **连续数周稳定工作**。

### 三阶段部署 + 三支柱

**场景扩展：** 工厂/仓库 → 酒店/养老/部分家庭（远程监控）→ 语言+示范快速学技能 → 家庭通用。

**七岁状态三支柱：** **长时记忆 + 可靠世界模型 + 可验证安全控制**（缺一则「动作漂亮、判断不稳」）。

### 八项工程工作（文内）

多模态真实生活数据（含失败/接管）→ 具身记忆 → 后果预测 WM → 高低层闭环 → **暂停优先** → sim–real 闭环 → 长时任务评测 → 隐私/权限/责任。

## 对 wiki 的映射

- [humanoid-cognitive-reliability-gap](../../wiki/concepts/humanoid-cognitive-reliability-gap.md)（本次升格主页面）
- [humanoid-eight-capabilities-technology-map](../../wiki/overview/humanoid-eight-capabilities-technology-map.md)（魔方 AI 八大能力 — 身体/智能/工程分层，与本页认知八能力 **不同框架**）
- [vla](../../wiki/methods/vla.md)（VLA 方法主线）
- [simulation-evaluation-infrastructure](../../wiki/concepts/simulation-evaluation-infrastructure.md)（长时评测基础设施）
- [data-flywheel](../../wiki/concepts/data-flywheel.md)（失败数据回流）
- [helix-25](../../wiki/entities/helix-25.md)、[gemini-robotics](../../wiki/entities/gemini-robotics.md)、[isaac-gr00t](../../wiki/entities/isaac-gr00t.md)、[pi07-policy](../../wiki/methods/pi07-policy.md)

## 可信度与使用边界

- 第三方科普（黄先生coin）；厂商数字与 Demo 以官方为准，不可从短视频反推完整架构。
- 「七岁状态」为 **有限自主比喻**，非儿童心理或安全认证标准。

## 当前提炼状态

- [x] 文章基础摘要填写
- [x] 初步 wiki 页面映射确认
