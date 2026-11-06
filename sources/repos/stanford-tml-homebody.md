# Stanford-TML/homebody

> 来源归档；2026-11-06 核查官方 README、`pyproject.toml`、`src/homebody/__main__.py` 和 `guides/architecture.md`。

- **项目页：** <https://tml.stanford.edu/homebody/>
- **仓库：** <https://github.com/Stanford-TML/homebody>
- **许可：** Apache-2.0；第三方组件遵循各自许可。
- **开放状态：** README 标注 SIM（10 月 5 日）、REAL2SIM（10 月 16 日）和 REAL（10 月 26 日）均已发布。REAL 需额外安装依赖及下载追踪权重。

## 官方运行入口

- `homebody --env src_kitchen --model astra`：MuJoCo 仿真与交互控制台。`pyproject.toml` 将命令映射至 `homebody.__main__:main`；`run` 创建 `MujocoBackend`、provider 与 `Session`。
- `homebody real2sim explore` 和 `homebody real2sim reconstruct`：探索记录与场景重建。
- `homebody run --backend real --real-config ... --memory ...`：连接已配置的真机服务，使用 `RealG1Backend`。
- 官方注明开源版采用 vanilla SuperOdometry，演示使用定位更强的内部版本，重建与真机结果可能不同。

## 关联

- [项目页归档](../sites/tml-stanford-homebody.md)
- [项目来源](../papers/homebody_tml_stanford_2026.md)
- [HomeBody 实体](../../wiki/entities/paper-homebody.md)
