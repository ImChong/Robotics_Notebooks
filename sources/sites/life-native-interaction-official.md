# LIFe — 官方研究演示项目主页（Native Interaction for Humanoid Robots）

> 一手资料归档；仅归纳已公开的官方项目页面，**不等同于未发布的论文全文**。

- **类型：** site / project-page / first-party qualitative demo
- **官方项目主页：** <https://qihanzhao.github.io/LIFe/>
- **站点原文件：** <https://github.com/QihanZhao/LIFe/blob/main/index.html>
- **官方 README：** <https://github.com/QihanZhao/LIFe/blob/main/README.md>
- **演示原视频文件：** <https://github.com/QihanZhao/LIFe/blob/main/assets/video/life-demo.mp4>
- **入库日期：** 2026-11-09
- **核对范围：** 已完整阅读官方仓库 README.md 与 index.html，核查仓库根目录与 assets 目录；没有逐帧播放/审阅 MP4，未获得论文或策略源码。
- **通读：** 已完成

## 身份与公开信息

- **项目名：** LIFe；官方页面标题为 *LIFe — Native Interaction for Humanoid Robots*，口号 *Move by teleop. Interact natively.*
- **截图副标题：** *Learning Native Interaction via Local Interaction Fields*；这行字幕来自用户给出的演示封面，尚未在已读官方网页正文找到“Local Interaction Fields”的技术定义。
- **作者（官方页署名）：** Qihan Zhao¹²*、Jiayang He²*、Yinhuai Wang¹²*、Ping Tan¹²；¹ HKUST，² SLAI，* 平等贡献。
- **真机：** 官方页面说明为 Unitree G1，并显示 VR 操作人员。
- **公开演示：** 官方页面嵌入一段 93 秒的 1080p60 真机视频，提供跳转章节：00:03 Catch、00:08 Continuous interaction、00:36 Exchange、01:00 Reorient。

## 官方愿景与演示任务

官方“THE IDEA”部分指出：漂亮的全身运动不等于可靠的物理交互，困难发生在物体接近、稳定接触、保持、适应变化和释放等接触瞬间。其愿景可概括为 **human guides motion + robot handles interaction → native interaction**，意在减少人工逐个接触细节的微操。

“THE MOMENTS”列有四类定性演示：**连续交互**（hold / carry / ball fake / throw）、**反应**（dynamic ball catching）、**适应**（chair / object reorientation）与**协作**（human–robot pass & catch）。这些均是官网的演示描述，不代表已经取得量化成功率。

## 开源与论文核对

| 核查项 | 结果 |
|---|---|
| 项目网页、前端脚本、图片及 MP4 | **已公开**，位于作者的 `QihanZhao/LIFe` 仓库 |
| 机器人策略训练 / 推理 / 部署源码 | **未公开**：README 明确说仓库仅含 public-facing project-page assets |
| 模型权重与 policy observations | **未公开**：README 明确排除 |
| 论文 / 技术报告 | **未公开**：README 称稿件尚未公开，未添加 preprint 链接 |
| ICRA / CoRL 接收情况 | **不可宣称**：README 明确未作接收声明 |
| Local Interaction Fields 结构、控制接口、损失函数、实验指标 | **待论文公开后验证** |

## Wiki 映射

- [LIFe 唯一项目实体](../../wiki/entities/life-native-interaction.md)
- [现有 HumanX 研究（不同项目）](../../wiki/entities/paper-hrl-stack-05-humanx.md)
