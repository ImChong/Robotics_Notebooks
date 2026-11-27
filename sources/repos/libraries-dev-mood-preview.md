# Libraries.dev Mood Recognition · 公开预览源码核对

- **仓库：** https://github.com/Jakubantalik/Libraries.dev
- **固定版本：** [d058179](https://github.com/Jakubantalik/Libraries.dev/tree/d058179b10adf2a2290957c4662b9c0ada1de98b)
- **项目页：** https://libraries.dev/mood-recognition
- **核对范围：** 核对固定提交的完整文件清单、README、模型许可及官网付款预览 React 入口；比对时间轴、UI 帧和 HUD 调用，并对实际源码中的纯函数运行 7 项断言。未调用 React 套件或模型。
- **通读：** 已完成

入口 [mood-recognition.tsx](https://github.com/Jakubantalik/Libraries.dev/blob/d058179b10adf2a2290957c4662b9c0ada1de98b/sites/home/src/mood-recognition.tsx) 用动画帧推进计时，读取预置指针路径和 `BEATS` 分值，再把状态交给 React、读数交给 `createHud`。预览用合成时间轴，不由访客鼠标实时推理。实测 7 项断言覆盖初始、提示变淡、突出条款、勾选恢复、付款帧、重置和四次模拟禁用点击。

仓库公开站点及多种 UI 效果的实现；README 明确当前模型及识别套件不随仓库分发。公开预览和需账号下载的推理套件不是同一个入口。

- **沉淀到 wiki：** [Mood Recognition Beta](../../wiki/entities/libraries-dev-mood-recognition.md)
