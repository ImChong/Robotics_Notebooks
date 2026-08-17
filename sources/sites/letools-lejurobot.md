# LeTools 官方网站与 AI 助手（letools.lejurobot.com）

> 来源归档（ingest）

- **类型：** 产品站 / 文档门户 / 在线助手
- **入口：** <https://www.letools.lejurobot.com/>
- **文档：** <https://www.letools.lejurobot.com/docs.html?type=learning>（Learning）· <https://www.letools.lejurobot.com/docs.html?type=skills>（Skills）
- **运营方：** 乐聚机器人（Leju Robotics）
- **入库日期：** 2026-08-17
- **抓取说明：** 以 **2026-08-17** 公开 HTML、`js/chat.js` 与文档引擎为准；产品参数与模型名单会随发版变化。

## 一句话

**LeTools** 是乐聚面向 **Kuavo 全尺寸人形** 的具身智能平台门户：宣称 **采集 → 训练 → 真机部署** 一条龙，并挂载站点内 **KuavoChat AI 助手**。

## 开源状态（步骤 2.5）

| 层 | 结论 |
|----|------|
| 产品站 / 文档站 | 公开可访问；文档由 `docs.html` 按 `type=learning\|skills` 拉取 Markdown |
| AI 助手 | **前端开源式静态脚本**（`js/chat.js`），后端为托管 Worker，**不是**可自托管的完整 agent 仓 |
| 训练栈 | [LeTools-Learning](https://github.com/LejuRobotics/LeTools-Learning) **GPL-3.0，已开源、可运行** |
| 技能/编排栈 | [letools_opensource](https://github.com/LejuRobotics/letools_opensource) **已开源、可运行**；GitHub API 无 SPDX license 字段，README 仅写「由乐聚维护」 |
| 整机 | Kuavo 为 **商业硬件**；站点不等于 CAD/固件全开源 |

## 门户主张（归纳）

| 区块 | 内容 |
|------|------|
| All In One Pipeline | Data → Deploy；采集、训策略、上真机 |
| Foundation Models | 统一框架微调 π 系、GR00T、LingbotVLA 等；宣传 **10+** 架构 |
| Tailored for Leju | 面向 Kuavo：硬件原生控制、出厂标定运动学、checkpoint→真机分钟级叙事 |
| 机型 | **Kuavo 4 Pro**（科研；单臂 7 DoF）· **Kuavo 5**（41 DoF、腰部、20 kg 负载）· **Kuavo 5W**（工业；续航 8 h、工作空间 0–2600 mm） |
| 使命三段 | 实验室性能 → 工业商用 → 家庭通用具身 |

## AI 助手（KuavoChat）

首页与文档页均加载 `css/chat.css` + `js/chat.js`。

| 项 | 截至 2026-08-17 |
|----|-----------------|
| 前端对象 | `KuavoChat.init({ workerUrl, model })` |
| 生产 Worker | `https://letools-chat-agent.huangrc1110.workers.dev`（注释中另有 `agent.kuavo.lejurobot.com`） |
| 默认模型 | `deepseek/deepseek-v4-flash` |
| 交互 | SSE 流式；会话写入 `sessionStorage`；建议问「LeTools 是什么 / 如何安装 Learning / rosbag 转换 / 支持哪些策略」 |
| 限额 | 前端按日缓存 remaining；默认上限叙事为 **20** 次/日 |
| 超时 | 整段 60 s 硬杀；单次 `read()` 20 s |

**边界：** 助手回答依赖托管后端与当日文档检索，不能替代仓库 README / Issues；密钥与 Worker 实现未开源。

## 为什么值得保留

- 把乐聚已有的 [整机官网](lejurobot.md) 与 [OpenLET 数据社区](openlet-openatom.md) 补上 **软件产品层**（训练胶水 + 原子技能编排）。
- 文档双栏（Learning / Skills）直接对应两个 GitHub 仓，避免把 IL 训练栈与行为树技能栈混成一个「SDK」。
- AI 助手是读者第一接触面，需写清 **托管、限额、非源码**。

## 对 wiki 的映射

- 升格：[wiki/entities/letools.md](../../wiki/entities/letools.md)
- 文档归档：[letools-docs.md](letools-docs.md)
- 仓库：[letools-learning.md](../repos/letools-learning.md)、[letools_opensource.md](../repos/letools_opensource.md)
- 硬件运营方：[wiki/entities/leju-robotics.md](../../wiki/entities/leju-robotics.md)
