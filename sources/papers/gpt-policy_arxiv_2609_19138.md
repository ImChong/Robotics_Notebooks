# GPT-Policy（上下文 VLM 机器人代理）

> 来源归档（ingest）

- **标题：** In-Context Robot Learning with VLM Agents
- **类型：** paper
- **原始链接：** <https://arxiv.org/abs/2609.19138>
- **项目页：** <https://cheng-haha.github.io/GPT-Policy/>
- **代码：** <https://github.com/cheng-haha/GPT-Policy>
- **机构：** Morphi Robot；上海创智学院；华中科技大学；复旦大学；湖南大学；香港中文大学；上海交通大学；武汉大学；东南大学；北京航空航天大学
- **作者：** Dongzhou Cheng*、Taoran Yi*、Ye Fang*、Xingwu Zhang*、Fan Feng*、Yixuan Li*、Gengxiong Zhuang*、Rongze Wang、Shuai Yang、Wei Song、Weizhi Xue、Minyan Wu、Jie Gui、Jiaqi Wang、Tong Wu†
- **入库日期：** 2026-09-18
- **一句话说明：** 固定通用 VLM + context compiler + 约束控制器闭环；部署时用人/机视频、目标图、自交互历史、人机交互等 **in-context** 信息，无梯度更新；GPT-6 Astra 真机十任务 ablation。

## 核心摘录（MVP）

### 1) 五类 context

- **摘录要点：** Human video（无 robot action）、Robot demo（可选对齐 action）、Target image、Self-interaction history、Human–robot interaction（指点/回合）。
- **对 wiki 的映射：**
  - [GPT-Policy](../../wiki/entities/paper-gpt-policy.md) — context 族
  - [GPT 6 Astra 具身策略评测](../../wiki/entities/paper-gpt-6-astra-embodied-policy.md) — 同模型族对照

### 2) Context compiler → VLM tool → 约束执行

- **摘录要点：** 交错图文 + tool schema 组成 VLM 输入；Cartesian adapter 做 IK 残差检查、路径采样、关节时序与 gripper；执行/拒绝反馈回写下一决策。
- **对 wiki 的映射：**
  - [GPT-Policy](../../wiki/entities/paper-gpt-policy.md) — 架构
  - [LLM 机器人控制接口](../../wiki/concepts/llm-robotics-control-interfaces.md)

### 3) 真机 ablation 要点

- **摘录要点：** Human video 在无 robot action 时仍提升 towel/notebook 成功率；contact-rich 任务（开瓶、插拔）**video + action** 优于仅 video；目标图任务 T-shape/fruit **3/3**；自历史与 HRI 任务亦满成功率（各 3 trials/condition，GPT-6 Astra）。
- **对 wiki 的映射：**
  - [GPT-Policy](../../wiki/entities/paper-gpt-policy.md) — 实验表

### 4) 开源状态（截至 2026-09-18）

- **摘录要点：** **已开源** — `cheng-haha/GPT-Policy`：`gpt-policy` CLI、ARX X5 / I2RT YAM adapter、`configs/default.json`；需自备 VLM API（GPT-6 Astra 等）。
- **对 wiki 的映射：**
  - [gpt-policy 仓库](../repos/gpt-policy.md)
  - [GPT-Policy 项目页](../sites/gpt-policy-cheng-haha-github-io.md)

## 当前提炼状态

- [x] 项目页 / README / PDF 已对齐
- [x] 步骤 2.5：**已开源**
- [x] wiki 映射：`wiki/entities/paper-gpt-policy.md`
