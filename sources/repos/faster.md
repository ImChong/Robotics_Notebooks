# alexanderswerdlow/faster 与 alexanderswerdlow/faster_vla

> 来源归档（repo）

- **标题：** FASTER 官方代码
- **Robomimic 实现：** <https://github.com/alexanderswerdlow/faster>
- **VLA 实现：** <https://github.com/alexanderswerdlow/faster_vla>
- **项目页：** <https://pd-perry.github.io/faster/>
- **论文：** <https://arxiv.org/abs/2604.19730>
- **机构：** Stanford University
- **入库日期：** 2026-10-07
- **一句话说明：** 同一论文的操作接口分成通用 Robomimic RL 与 π0.5 VLA 复现两部分。

## 开放程度

| 仓库 | 覆盖 |
|------|------|
| faster | Robomimic 操作任务；包含在线与 batch-online FASTER-EXPO / FASTER-IDQL 脚本 |
| faster_vla | π0.5 VLA 代码；README 说明训练与 LIBERO 环境分开运行，通过 UNIX socket 传观测和动作 |
| 依赖 | VLA 实现依赖 OpenPI 子模块、LIBERO 与相应数据/模型配置，按 README 锁定版本 |

## 对 wiki 的映射

- [paper-faster](../../wiki/entities/paper-faster.md)
- [论文摘录](../papers/faster_arxiv_2604_19730.md)
