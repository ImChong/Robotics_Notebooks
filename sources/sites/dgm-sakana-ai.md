# Darwin Gödel Machine — Sakana AI 官方项目介绍

> 来源归档（ingest）｜独立作者/机构一手资料，不是转载

- **类型：** 官方项目介绍 / research blog
- **官方地址：** <https://sakana.ai/dgm/>
- **论文：** <https://arxiv.org/abs/2505.22954v3>
- **官方代码：** <https://github.com/jennyzzt/dgm>
- **核查日期：** 2026-11-10
- **通读：** 已完成
- **开放状态：** 核对到公开官方 GitHub 代码与 Apache-2.0 LICENSE；README 有 Docker 与 Python 主入口、依赖、SWE-bench / Polyglot 使用说明及实验日志链接。不能把发布代码等同于已有充分安全的生产系统。
- **对 wiki 的映射：** [Darwin Gödel Machine](../../wiki/entities/paper-darwin-godel-machine.md)

## 官方资料要点

1. **目标：** 在不更新底层基础模型权重的情况下，让 coding agent 迭代改写自己可执行的代码、工具与工作流，经验式检查对编程任务的提升。
2. **核心结构：** 不把进化压成一条单一贪心路径；维护包含多个分支的版本档案，给未来自改进保留可能有用的 stepping stone。
3. **实验证据：** 编码 benchmark 受任务集、语言模型、评测预算影响；论文 SWE-bench Verified 的 20.0%→50.0% 及全量 Polyglot 的 14.2%→30.7% 应按原论文实验协议阅读。
4. **风险：** 代码修改可以产生意外行为，得分可能出现 objective hacking；实际系统必须限制权限、沙箱运行、记录 lineage 并人工监督。

## 与官方实现交叉核对

- [README](https://github.com/jennyzzt/dgm/blob/main/README.md)：许可证、启动方式和安全提醒。
- [DGM_outer.py](https://github.com/jennyzzt/dgm/blob/main/DGM_outer.py)：父节点采样、并行 self-improve、archive 更新。
- [self_improve_step.py](https://github.com/jennyzzt/dgm/blob/main/self_improve_step.py)：在 Docker 内恢复父代代码、自改生成补丁、调用评测器。
- [论文 v3 HTML](https://arxiv.org/html/2505.22954v3)：数值、消融、Appendix H 目标劫持案例。

**复现声明：** 已核对官方文档与关键代码路径；未本地执行完整的 SWE-bench 或 Polyglot 进化实验。
