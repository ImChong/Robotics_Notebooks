# badminton-pipeline-repro

- **类型：** repo
- **链接：** https://github.com/ychenfen/badminton-pipeline-repro
- **入库日期：** 2026-11-01
- **开放状态：** 公开源码；仓库注明代码采用 MIT，模型权重和视频各按来源许可使用。
- **作用：** 羽毛球广播视频的球与球员追踪、球场映射和视频叠加。
- **核查范围：** 官方 README、`run_all_mac.sh`；未实际运行。

脚本依次运行 TrackNet 推理生成视频及 CSV、YOLOv8s-pose 与 ByteTrack 叠加、最后生成特效视频。球场角点需手动标注或提供坐标；README 将自动回合分割和热力图列为后续任务。该仓使用 ByteTrack，不能据此断定其就是截图所示 BoT-SORT 系统。

## 待完成

通读 HANDOVER 和各阶段实现，编写项目实体、项目流程图及源码运行时序图；核验截图与该仓的关系。
