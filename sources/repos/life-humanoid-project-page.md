# QihanZhao/LIFe — 官方项目演示页面仓库

> 一手仓库归档。该仓库**不是** LIFe 机器人控制、训练或推理源码。

- **类型：** repo / public project-page assets / HTML, CSS, JavaScript, video
- **官方仓库：** <https://github.com/QihanZhao/LIFe>
- **官方网页：** <https://qihanzhao.github.io/LIFe/>
- **入库日期：** 2026-11-09
- **核对范围：** 完整阅读仓库 README.md 和 index.html、检查仓库根目录及 assets/ 的文件列表；未分析或运行机器人策略（仓库中不含此类程序）。
- **通读：** 已完成

## 文件与运行入口

根目录公开 `README.md`、`index.html`、`deploy.sh`、`assets/`，其中 `assets/` 包含网页脚本、样式、照片与 `assets/video/life-demo.mp4`。README 推荐使用 `python3 -m http.server 8000` 在本机预览**静态站点**；它不是人形机器人算法入口。

## 开放范围

README 明确说明：“This repository contains only public-facing project-page assets.” 并明确不包括训练代码、模型权重、policy observations、实现细节和论文手稿。官网只提供定性实机演示和研究愿景。不能把网页的 `assets/main.js` 当作机器人运行时的模块，也不能从它推导 Local Interaction Fields 的数学形式。

## 关联与后续

- [官方页面归档](../sites/life-native-interaction-official.md)
- [LIFe 唯一 wiki 实体](../../wiki/entities/life-native-interaction.md)
- 论文与算法仓库开放后，重新核验源码入口与控制、训练链，届时才能绘制真正的源码运行时序图。
