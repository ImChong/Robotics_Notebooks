# LIFe — Learning Native Interaction via Local Interaction Fields｜截图及官方一手资料交叉核验

> **入库状态：已找到官方项目网页与演示资源；论文仍未公开。** 标题来自用户提供的短视频截图，“Local Interaction Fields” 算法结构尚不能核验。

- **类型：** video / 截图线索 + 官方视频入口
- **截图中的英文完整标题：** LIFe — Learning Native Interaction via Local Interaction Fields
- **中文暂译：** 通过局部交互场学习原生交互
- **截图账号：** QihanZhao（账号文字；官方 GitHub 仓库也由 `QihanZhao` 发布）
- **展示文案：** “Move by Teleop. Interact Natively.”
- **官方项目网页：** <https://qihanzhao.github.io/LIFe/>
- **官方 GitHub：** <https://github.com/QihanZhao/LIFe>
- **官方演示文件：** <https://github.com/QihanZhao/LIFe/blob/main/assets/video/life-demo.mp4>
- **原社交短视频 URL：** 尚未知（用户截图未提供固定地址）
- **论文 / arXiv ID：** 尚未公开，不推定编号
- **入库日期：** 2026-11-09
- **核对范围：** 对照用户截图辨认标题与宣传文案，完整阅读官网源文件 `index.html` 与 `README.md`；未取得论文全文，未观看社交平台原视频，也未对 MP4 逐帧分析。
- **通读：** 未完成

## 已核实的一手事实

- 官网标题为 **LIFe — Native Interaction for Humanoid Robots**，口号为 *Move by teleop. Interact natively.*。完整 “Learning Native Interaction via Local Interaction Fields” 是截图副标题，在已读官网正文中尚无正式算法定义。
- 作者署名 Qihan Zhao、Jiayang He、Yinhuai Wang、Ping Tan；标注 HKUST、SLAI 和前三位平等贡献。
- 官网标注 **Unitree G1**，嵌入一段 **93 秒、1080p60** 真机演示，网站提供 Catch、Continuous interaction、Exchange、Reorient 四个章节。
- 官方网站展示的案例含球类连续交互、动态接球、人机传接球和椅子位置调整。
- 官方 README **明确**：仓库只有项目页面素材，不含机器人训练/推理代码、模型权重、policy observations、实现细节及论文手稿；没有 ICRA/CoRL 录用声明。

## 尚未核实

- Local Interaction Fields 的数学形式、观测/动作空间与训练目标；
- 遥操作与接触自主控制的运行时分工、推理频率、机器人输入输出；
- 实验成功率、基线、消融、泛化与安全可靠性；
- 论文全文、会议版本与机器人模型/代码许可。

## 归档映射与后续

- [官方项目页一手归档](../sites/life-native-interaction-official.md)
- [官方网站仓库归档](../repos/life-humanoid-project-page.md)
- [唯一 wiki 实体：LIFe（draft）](../../wiki/entities/life-native-interaction.md)
- 论文和算法源码发布后，先通读全文和运行入口，再绘制训练/控制架构与源码时序图、更新评测和节点健康度。
