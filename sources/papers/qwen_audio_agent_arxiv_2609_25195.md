# Qwen-Audio-Agent Technical Report（arXiv:2609.25195）

> 来源归档

- **标题：** Qwen-Audio-Agent Technical Report
- **类型：** paper（arXiv technical report）
- **链接：** https://arxiv.org/abs/2609.25195
- **PDF：** https://arxiv.org/pdf/2609.25195
- **代码：** https://github.com/QwenAudio/qwen-audio-agent
- **文档 / 项目页：** https://qwenaudio.github.io/qwen-audio-agent/
- **机构：** Alibaba Token Foundry, Alibaba Group
- **入库日期：** 2026-10-01
- **一句话说明：** 开源 **全双工语音 harness**：前台 Frontend Agent 管对话与路由，后台 Backend Agent 异步执行委派任务；Orchestration Runtime 独立管理任务生命周期、打断与结果投递，并支持跨会话记忆与环境事件；座舱 benchmark 上 **mixed** 路由任务成功率 **91.04%**，相对全前台 / 全委派降低平均执行延迟。
- **沉淀到 wiki：** [Qwen-Audio-Agent](../../wiki/entities/paper-qwen-audio-agent.md)

---

## 核心贡献（摘要 / 正文归纳）

1. **Foreground–background 架构：** Frontend Agent 全双工对话 + 选 **直接工具** 或 **spawn_thinking 委派**；Backend Agent 在独立 context 中多步执行；客户端负责采集、播放与本地动作。
2. **异步任务协调：** 任务记录含 objective / status / progress / outputs / pending requests；**语音打断 ≠ 任务取消**；**执行完成 ≠ 立即播报**，结果可合并窗口后投递。
3. **可插拔适配：** 前台 Realtime Provider、后台 ACP / A2A / 自定义 Agent；Table 1 强调相对 gpt-realtime / GPT-Live / Gemini Live 的 **跨厂商前台** 与 **开源编排运行时**。
4. **环境事件与记忆：** 统一 event 协议（如座舱 UI 改目的地可静默入上下文）；会话内规则 + 用户偏好 + 长期记忆，异步检索与边界修订策略。
5. **三类落地：** 桌面办公、智能座舱、语音客服（preview-and-commit 授权）；另报数字人与智能硬件部署。
6. **座舱评测（134 cases）：** Qwen Audio 3.0 Realtime Plus 前台 + qwen3.8-max 后台；**mixed execution** 任务成功率 **91.04%**（direct **72.39%**，all delegated **80.60%**）；匹配成功 turn 上 mixed 平均任务执行延迟 **4.729 s**（相对 direct **−26.73%**，相对 all delegated **−30.91%**）。

---

## 对 wiki 的映射

- 主实体页：[paper-qwen-audio-agent.md](../../wiki/entities/paper-qwen-audio-agent.md)
- 交叉更新：`wiki/methods/humanoid-voice-interaction.md`（全双工 + 异步工具 / 后台 Agent 编排参考）

---

## 摘录要点（维护者可据 PDF 深化）

- **spawn_thinking：** 前台提交自包含 objective（含约束与输入引用）；前台对话历史 **不会** 自动整包传给后台。
- **协议：** Backend 侧 Agent Client Protocol (ACP)、Agent2Agent (A2A) 或自定义接口。
- **对比表（Table 1）：** Qwen-Audio-Agent 唯一勾选 **开源编排运行时** 与 **跨 provider 前台** 等组合能力（相对闭源 Realtime API 产品）。
- **开源结论（论文脚注 + GitHub）：** 实现 **已开源**；npm 包 `qwen-audio-agent`，Gateway CLI `qwenaudio`。
