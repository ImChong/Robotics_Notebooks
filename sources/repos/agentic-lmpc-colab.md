# LMPC 官方教学 Colab 静态核查

- **代码：** https://colab.research.google.com/drive/1YcRN_kklw3cVVJNvgK_IEV6nDce9EJWK
- **官方入口：** [项目页归档](../sites/agentic-lmpc-2402-11450.md)；[论文归档](../papers/agentic_2402_11450.md)。
- **快照：** 2026-11-16 获取公开 Notebook JSON，SHA-256 `6bb19d85a742132fc0440808aabd66178f0ef83d5e10dec197b64d964b8ef46a`，30 个单元；Apache-2.0 声明。
- **核对范围：** 全部 markdown 和 code cells，含 SimpleParticleNav、ParticleAgent、code_to_actions、collect_data、JSONL 格式化、OpenAI 文件/微调 API、LMPC 与评估调用；不含运行验证。
- **通读：** 已完成

## 开放范围

Notebook 明示 illustrative purposes only，是 2D 粒子导航与模拟用户噪声的教学实现。用历史 gpt-3.5-turbo-1106 模型；不是论文的 PaLM 2、MJPC 五本体及真机部署全实现。未运行任何 API、上传文件或创建付费训练。

## 实际入口和论文差异

1. collect_data 调用环境 reset，再按 policy 生成代码；code_to_actions 通过 exec 写入 ParticleAgent.delta，env.step 返回反馈和成功标志。
2. 单元 18 将全部会话按任务分 train/test、追加 success 标签并写 JSONL，没有只保留成功会话的过滤；而论文主设置仅用成功会话。顶部 TOP_USERS 手工指定零噪声模拟用户，不能冒充论文按难度评分自动选前 25% 的机制。
3. 单元 20 调用 files.create、fine_tuning.jobs.create 并轮询，获得 fine_tuned_model。
4. LMPC 函数采样四条未来消息，检测 success: True，尝试选短候选并返回第一条代码。无成功样本时默认最后一条候选；论文则八候选且无终止样本时随机回退。源码将 best_rollout 初始化为最后一次采样，即使该条未成功也可能作为长度比较基准；这是教学实现的选择边界，不能据此声称实现了严格最短成功路径。

## 对 wiki 的映射

[LMPC 独立详情与 Notebook 时序图](../../wiki/entities/paper-lmpc-2402-11450.md)。
