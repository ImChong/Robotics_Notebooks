# ACAP Rollout Dataset / ACAP 真机执行记录

- **数据卡：** https://huggingface.co/datasets/zxhezexin/acap-rollout-dataset
- **目录元数据：** [Hugging Face Dataset API](https://huggingface.co/api/datasets/zxhezexin/acap-rollout-dataset)
- **核对版本：** `d8062d5c1873ebdabab7626c105b1a4493f587db`（2026-11-26 核对）
- **许可与访问：** 数据卡标 CC BY 4.0；Hub 标记为人工审批门控，数据卡可见但文件请求未经授权返回 401。本次未申请权限、下载轨迹或验证归档数据字节。
- **核对范围：** 通读发布方可见数据卡、字段与下载说明；用目录文件数核验两个划分，未读取门控文件内容或运行其程序。
- **通读：** 已完成

| 划分 | 数量 | 说明 |
|---|---:|---|
| `task_coverage` | 72 | 25 个任务/平台设置含独立、顺序共享空间和协作；Fixed 用于其中协作任务 |
| `repeated_trials` | 90 | 六任务 × 三策略 × 五次；Direct、Fixed Delegation、ACAP |

目录含两份 Parquet 索引、`manifest.jsonl`、`SHA256SUMS` 及 162 个轨迹归档；数据卡报告压缩共 136.85 GiB。原始模式标签为 `singleton` / `static` / `adaptive`，分别对应上述三策略。索引可通过任务、平台、依赖类别等字段筛选，并通过 `archive_path` 找到对应文件；`outcome_basis` 记录结果标签依据。各归档可用字段与文件不尽相同，可能包括任务指令、观测与反馈、生成程序、消息和控制事件。回放可以省略等待，不能从视频长度推断评价耗时。记录的代码和环境路径不等同于可直接部署的控制器。

- **论文：** [arXiv:2611.31205 核对](../papers/acap_arxiv_2611_31205.md)
- **Wiki：** [ACAP 论文节点](../../wiki/entities/paper-acap.md)
