# Hugging Face Space：robotaemoon/CoRe

> 来源归档（ingest）

- **标题：** CoRe live demo（Hugging Face Space）
- **类型：** site / hosted-demo
- **URL：** <https://huggingface.co/spaces/robotaemoon/CoRe>
- **入库日期：** 2026-08-15
- **配套仓库：** <https://github.com/tmjeong1103/CoRe> — 归档见 [`sources/repos/core_retarget.md`](../repos/core_retarget.md)

## 一句话摘要

CoRe v0.1.0 的官方浏览器体验：上传 Kimodo `.npz` 或 GEM-X `.pt`（或一键跑捆绑示例），选择 11 台人形之一，预览终态运动并下载安全的 `robot_motion.npz` 与 manifest。

## 公开信息要点（截至入库日）

- **后端：** 编译 C++ MuJoCo 核；生产向 Docker Space（有界队列、结果过期）
- **本地等价：** `pip install -e ".[web]"` 后 `core-retarget serve`（默认 `127.0.0.1:8000`，单任务串行）
- **部署说明：** 仓库 `docs/huggingface-space.md`
- **不改变算法：** Web 适配器只加上传校验、进度流与制品白名单，调用同一 `run_retarget_pipeline()`

## 对 wiki 的映射

- [`wiki/entities/core-retarget.md`](../../wiki/entities/core-retarget.md)
