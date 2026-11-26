# ICLWorld：官方任务与演示资产

- **数据：** [Aoraku/ICLWorld](https://huggingface.co/datasets/Aoraku/ICLWorld)；[本次核对版本 99cacf7](https://huggingface.co/datasets/Aoraku/ICLWorld/tree/99cacf7147e18eb16f983f048765afc76b20f36a)。
- **核对范围：** 数据卡、发布文件索引、元数据字段与发布 README；未拉取全部视频或加载仿真。
- **通读：** 已完成

- 统一的 `metadata.jsonl` 有 480 条任务 × 规则记录，即 160 个任务 × A/B/C；论文的 630 段由 330 段 Web/OS 演示和 150 对真实/仿真机器人演示组成；数据卡另列 101 段级别专用机器人输入，故发布卡列 401 段机器人视频，不是 401 段真实录制。视频数也不是任务数或评测次数。
- `asset/cua/` 包含 Web/OS 素材；`asset/rua/` 包含机器人录制及仿真资产。按数据卡与代码仓提供的原生目录结构准备；本次访问时 Hugging Face 通用表格 Viewer 未能识别该发布结构，不等于文件不可下载。
- 数据卡未标明数据整体许可；仓库 MIT 不等于视频媒体及第三方环境资产均可再分发。发布的元数据和 QA 答案仅交给评测方，不放进被测智能体上下文。

- **Wiki：** [ICLWorld](../../wiki/entities/paper-iclworld.md)
