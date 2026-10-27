# FastHarness 官方代码

> 来源归档（repo）

- **仓库：** <https://github.com/FastHarness/fast-harness>
- **项目页：** <https://fastharness.github.io/>；[站点归档](../sites/fastharness.md)
- **论文：** <https://arxiv.org/abs/2610.31427>；[论文归档](../papers/fastharness-experience-driven-review-scheduling_arxiv_2610_31427.md)
- **核查日期：** 2026-10-27
- **开放状态：** 公开代码，MIT 许可证。仓库包含离线符号演示与 RoboDojo 仿真集成；不包含 RoboDojo 或 VLA 策略服务器，需自行提供运行时、模型和评审接口。

## 运行入口

- `python -m fast_harness demo --output /tmp/fh_demo --memory-dir /tmp/fh_demo_mem --episodes 6 --mode compare`：离线对比。
- `python -m fast_harness live`：连接外部策略服务器、RoboDojo 和评审模型；具体参数见 README。
- `python -m unittest discover -s tests -v`：运行离线测试。

## 关联知识页

- [FastHarness](../../wiki/entities/paper-fastharness-experience-driven-review-scheduling.md)
