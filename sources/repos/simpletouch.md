# SimpleTouch 官方代码

- **仓库：** <https://github.com/ccyydd/SimpleTouch>
- **论文：** <https://arxiv.org/abs/2610.02784>；[论文归档](../papers/simpletouch_arxiv_2610_02784.md)
- **项目页：** <https://simpletouch-robot.github.io/>；[项目归档](../sites/simpletouch.md)
- **权重：** <https://huggingface.co/cyang203912/SimpleTouch>
- **许可：** Apache-2.0；第三方组件另有许可。
- **入库日期：** 2026-10-27

## 开放与运行入口

仓库公开，README 提供训练及 UniVTAC 评测流程。Linux、Python 3.11 和 NVIDIA GPU 用于策略环境；仿真评测另需 UniVTAC、Isaac Sim 4.5 等依赖。训练示例每任务取 50 条演示，默认 8 GPU、20000 步；先运行 `scripts/univtac/recompute_norm_stats.py` 和 `scripts/univtac/cache_t3_latents.py`，再运行 `scripts/simpletouch/train_univtac.sh`。评测入口为 `scripts/simpletouch/evaluate_univtac.sh`。README 提供检查点链接，但本次无法独立访问权重页验证下载。
