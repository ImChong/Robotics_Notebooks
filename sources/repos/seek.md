# SEEK Agentic Robot Manipulation 官方仓库

- **类型：** 源码
- **链接：** https://github.com/ZeyangGuan/SEEK-Agentic-Robot-Manipulation
- **核对提交：** [318cef0](https://github.com/ZeyangGuan/SEEK-Agentic-Robot-Manipulation/tree/318cef0e3c22a8e4a72a980df9a65af807c36bfb)
- **许可证：** MIT
- **论文：** [arXiv:2611.33915v1](../papers/seek_arxiv_2611_33915.md)
- **项目主页：** [项目页归档](../sites/seek.md)

README 与 `docs/SETUP.md` 给出仿真运行环境。启动器 `runtime/seek/run.py` 启动策略服务，再启动仿真客户端；`runtime/seek/rollout.py` 内可查 `reset`、`update_obs`、`get_action`、运动查询与 `finalize` 的 RPC 顺序；`runtime/tools/review.py` 负责回合后的复查。逐回合日志在 `evaluation/k5-frozen-seed2/` 与 `evaluation/astra-direct-seed2/`。仅静态核对，未装 Isaac 或调用模型端点。
