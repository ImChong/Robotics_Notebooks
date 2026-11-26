---
type: repo
url: https://github.com/Tripo-Developer/tripo-python-sdk
title: "Tripo Python SDK — 官方 API v3 客户端"
added: 2026-11-26
---

# Tripo Python SDK 官方代码归档

- **仓库：** [Tripo-Developer/tripo-python-sdk](https://github.com/Tripo-Developer/tripo-python-sdk)，[核对快照 aa94035](https://github.com/Tripo-Developer/tripo-python-sdk/tree/aa94035f66cbefbd953ff9353078a769e6f06b48)，版本 1.0.0。MIT 只涵盖客户端代码。
- **核对范围：** README v3 迁移和快速开始、LICENSE、pyproject、文生 3D 示例，以及客户端的提交、轮询、任务状态解析、下载和官方迁移公告。
- **通读：** 已完成

## 可运行范围

- [examples/text_to_model.py](https://github.com/Tripo-Developer/tripo-python-sdk/blob/aa94035f66cbefbd953ff9353078a769e6f06b48/examples/text_to_model.py) 调用 TripoClient 提交任务、等待成功后建目录下载。
- [tripo3d/client.py](https://github.com/Tripo-Developer/tripo-python-sdk/blob/aa94035f66cbefbd953ff9353078a769e6f06b48/tripo3d/client.py) 中 text_to_model 调用 POST /v3/generation/text-to-model；wait_for_task 轮询 GET /v3/tasks/{task_id}；download_task_models 按 URL 去重，同 URL 的 model 与 pbr_model 可指向同一文件。
- 真实调用需独立 API 凭证与额度。示例未显式指定模型，而该源码仍默认 v2.5-20250123；[下线公告](https://developers.tripo3d.ai/en/blog/legacy-model-deprecation) 要求于 2026-12-01 前迁移。新调用显式指定 [官方推荐版本](https://developers.tripo3d.ai/en/docs/models-and-versions)，如 v3.1-20260211。
- 此仓库不能自托管 [Web Studio](../sites/tripo-studio.md) 或生成服务。详见 [Wiki](../../wiki/entities/tripo-studio.md)。
