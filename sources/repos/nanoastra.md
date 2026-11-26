# nanoAstra | 官方仿真执行与评测实现

- **代码：** [Hugo-AGI/nanoAstra @ 5a99fb7](https://github.com/Hugo-AGI/nanoAstra/tree/5a99fb76664c51df95843fda4fa6cbd69e7edc12)，MIT；RoboTwin 子模块固定于 `96c1feab536306b50c26af200044fcdf126e8904`，独立许可。
- **权重：** [HF @ 355ab76](https://huggingface.co/Hugo-AGI/nanoAstra/tree/355ab76f3705fc9f3304cd64668662c8a0d6590f)：公开模型卡标 Apache-2.0，含配置、模板与 4 个 safetensors 分片；未下载权重或实跑 vLLM/RoboTwin。
- **论文与项目：** [arXiv v1](../papers/nanoastra_arxiv_2611_29338.md) · [官方项目页](../sites/nanoastra.md)
- **核对范围：** 2026-11-26 核对 README、serve/eval/summarize、agent、prompt、steps、streaming 和环境适配器；在 CPU 运行官方单测（13 个通过），核对仓内 50 份任务 seeds 文件。
- **通读：** 已完成

- `scripts/serve.sh` 启动 vLLM；`scripts/eval_robotwin.py` 选任务和模式，经 `Agent.run_episode` 收集图像、本体状态、历史与回执，产出 trace 和结果。
- `steps.py` 解析双臂结构及参数，`robotwin/env.py` 将语义步转为底层规划器的目标位姿与夹爪设置。发布实现除 move、rotate、gripper、wait，还识别 point、home 与独立 done 标记。
- `streaming.py` 用 H20 的首 token、输入长度和输出 token 延迟拟合来排程仿真时钟，不直接把本机请求耗时当测试口径。剩余不超过两步时复用已承诺前缀，跳过重复执行。
