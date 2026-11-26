# Video Generation and Editing Workflow · 视频生成与编辑流程：一手资料复核

- **类型：** 方法一手资料归档
- **核对日期：** 2026-11-26
- **通读：** 已完成
- **核对范围：** 以下所列文档、接口和实现范围；未运行实机或评测。
- **对应页面：** [方法页](../../wiki/methods/video-generation-workflow.md)

## 一手资料

- [Wan2.1 官方仓库 README](https://github.com/Wan-Video/Wan2.1)：阅读 Quickstart、Model Download、T2V、I2V、首尾帧与 VACE 推理、Manual Evaluation；未运行权重。

## 交叉核对

- 先按任务锁定条件和权重。Wan2.1 的公开示例区分 T2V、I2V、首尾帧生成和 VACE 编辑；后者可传参考图、原视频和遮罩，部分编辑任务需先生成额外条件。不同权重的官方示例分辨率不同。
- 看起来合理的视频不提供可执行动作或动力学真值；遮罩外也可能出现闪烁或身份漂移。官方发布的推理示例和权重不是对任意任务、许可证或数据来源的担保，本页未运行模型。
