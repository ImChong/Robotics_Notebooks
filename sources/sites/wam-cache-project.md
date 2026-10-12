# WAM-Cache 项目页（dingkai0302.github.io/wam-cache）

> 来源归档

- **标题：** WAM-Cache
- **类型：** site（项目页，GitHub Pages）
- **URL：** <https://dingkai0302.github.io/wam-cache/>
- **论文：** <https://arxiv.org/abs/2610.11401>
- **代码：** <https://github.com/DingKai0302/wam-cache>
- **入库日期：** 2026-10-12
- **一句话说明：** 免训练的 WAM 视频 DiT prefill 加速：分层 KV 缓存 + 稀疏刷新集（潜变量惊异、动作专家交叉注意力、年龄界），在 Fast-WAM 上降低 32–42% FLOPs。

## 页面结构（维护索引）

| 区块 | 内容要点 |
|------|----------|
| 方法 | 刷新集三准则；“动作专家看哪里”比“视觉哪里变了”更重要 |
| 结果 | RoboTwin 2.0 1.053→0.615 TFLOPs、90.16%→88.36%；LIBERO 0.859→0.582 TFLOPs、96.50%→95.85%；较大预算 90.08%/96.50% |
| 链接 | 论文、GitHub、BibTeX |

## 开源核查（2026-10-12）

- 项目页有 Code 链接且无“即将发布”提示，但目标仓库 DingKai0302/wam-cache 只有 docs/（项目页）与 README.md（2 次提交），README 写 “Code coming soon”，无脚本、权重或许可证。结论：待发布（占位仓库）。
