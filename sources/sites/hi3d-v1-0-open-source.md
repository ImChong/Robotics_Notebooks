# Hi3D v1.0 官方开源项目页

- **类型：** 项目页 / 开源发布页
- **入口：** <https://www.hi3d.ai/hi3d-1-0-open-source>
- **主体：** Hi3D 官方团队
- **代码：** <https://github.com/hi3d-official/Hi3Dv1.0>（推理代码）
- **模型权重：** <https://huggingface.co/hi3d-official/Hi3Dv1.0>
- **在线体验：** <https://huggingface.co/spaces/ilcve21/Sparc3D>（由官方 GitHub README 列为 Live Demo）
- **收录日期：** 2026-10-23

## 一句话

Hi3D v1.0 是基于 Sparc3D 的高分辨率三维几何细化模型：将参考图像与同一物体的粗网格结合，逐级细化为更高分辨率的网格几何。

## 开源与使用边界核查（2026-10-23）

官方 GitHub 仓库公开推理代码、运行示例和文档，模型权重托管于 Hugging Face。仓库明确说明训练代码和论文评测工具不包含在发布内容中。README 报告训练使用约 200 万精选 3D 资产，但未提供对应训练集下载入口。

代码与权重适用仓库中的 **Hi3D Open License Agreement Version 1.0**。官方 README 说明协议允许商业使用，同时对规模化商业部署设定额外要求：若月活用户超过 200,000，或月收入达到 1,000,000 美元，须在 60 日内通知许可方并申请单独书面许可；商业集成需展示 **Powered by Hi3D 1.0**。协议还包含声明和可接受使用政策等要求，具体解释以仓库许可证正文及第三方许可说明为准。公开下载不等于无条件商用。

## 在线体验说明

GitHub README 提供的 Live Demo 指向 Hugging Face 的 `ilcve21/Sparc3D` Space。其页面名称仍是 Sparc3D，运行状态、底层模型版本和访问条件可能随部署调整；归档该入口不代表保证其当前免费、可用或已切换到 Hi3D v1.0。使用前请核对 Space 页面说明。

## 对机器人与具身研究的价值

模型可用于给粗网格补充局部几何细节，作为视觉资产制作或仿真道具准备流程的一环。但它输出的是重建/补全网格，不会自动提供机器人 URDF/MJCF 关节、准确物理参数或接触验证；进入物理仿真前仍需检查尺度、碰撞几何、拓扑与动力学属性。

## 参考链接

- 官方发布页：<https://www.hi3d.ai/hi3d-1-0-open-source>
- 官方代码仓库：<https://github.com/hi3d-official/Hi3Dv1.0>
- 官方模型权重：<https://huggingface.co/hi3d-official/Hi3Dv1.0>
- 在线 Space：<https://huggingface.co/spaces/ilcve21/Sparc3D>
- Sparc3D 基础论文：<https://arxiv.org/abs/2505.14521>
