# emrumo/swingingdoor：Swinging Door Python 参考实现

> 来源归档：社区实现（非算法作者/厂商官方代码）

- **仓库：** <https://github.com/emrumo/swingingdoor>
- **文件入口：** [compressor.py](https://github.com/emrumo/swingingdoor/blob/main/compressor.py)、[README.md](https://github.com/emrumo/swingingdoor/blob/main/README.md)、[LICENSE](https://github.com/emrumo/swingingdoor/blob/main/LICENSE)
- **类型：** repo / Python / online univariate time-series compression
- **核查日期：** 2026-10-10
- **许可证：** MIT（以仓库 LICENSE 为准）
- **维护/规模快照：** 本次核查仓库展示 2 commits、无 release；作为学习与算法交叉核对用的小型参考实现，不是工业级或 Bristol/AVEVA 官方发行版。
- **一句话说明：** Python 实现在线单变量时间序列压缩，使用归档锚点、快照和偏差参数维护上下斜率锥，并提供 compMin / compMax 时间检查。

## 实现结构摘录

- SwingingDoorState 保存当前 snapshot 与上下界函数；更新后仅收窄可行斜率区间。
- SwingingDoorArchiver 保存输出点并提供时间戳/信号值读取。
- SwingingDoor.compression_test 先做首次点/外部强制输出、compMin/compMax 检查，再执行 cone test；当新点超出当前锥时输出前一 snapshot。
- README 声称线性插值重建误差受用户指定 compression deviation 限制。对具体应用仍需用自己的边界测试核对时间戳重复、NaN、乱序、数值极值和 max-interval 行为。

## 开源核查

仓库根目录提供 MIT License；README 列出 NumPy、matplotlib、celluloid 依赖。该许可只适用于这个代码仓库，不能代表 PI Server、原始专利或其他 SDT 实现的授权状态。仓库未提供 release，使用前需自行验证并固定 commit。

## 对 wiki 的映射

- [Swinging Door Trending（摆动门趋势压缩）](../../wiki/concepts/swinging-door-trending-compression.md) — 算法状态、实现参数和适用边界。
- [原始专利 US4669097A](../patents/bristol_swinging_door_us4669097a.md) — 原始技术溯源；本仓库只是独立的社区实现。

## 参考来源

- [GitHub 仓库 README](https://github.com/emrumo/swingingdoor/blob/main/README.md)
- [Python 实现 compressor.py](https://github.com/emrumo/swingingdoor/blob/main/compressor.py)
- [MIT License](https://github.com/emrumo/swingingdoor/blob/main/LICENSE)
