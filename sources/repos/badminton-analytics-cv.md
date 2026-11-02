# BadmintonAnalyticsCV

- **类型：** repo
- **链接：** https://github.com/Siddharth194/BadmintonAnalyticsCV
- **入库日期：** 2026-11-01
- **开放状态：** 公开源码；权重可用性及授权需另核查。
- **作用：** README 介绍结合 TrackNet 与 YOLO，生成精彩片段及比分叠加。
- **核查范围：** README、`pre_predict.py`、`testing.py`；未实际运行。

`pre_predict.py` 分段预测并输出球位置 CSV 与 `predicted.bin`；`testing.py` 读取预测结果，调用辅助函数处理回合并叠加比分。README 和已查源码尚不足以证明它就是截图中的完整系统。

## 待完成

核查 `predict.py` 及回合、比分辅助函数，验证依赖与运行入口；编写项目实体、流程图与源码运行时序图。
