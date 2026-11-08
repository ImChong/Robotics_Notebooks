# FaceFusion — 官方产品入口、安装文档与处理器文档

- **类型：** site / official documentation
- **项目入口：** https://facefusion.io/
- **文档首页：** https://docs.facefusion.io/
- **代码：** https://github.com/facefusion/facefusion
- **核对日期：** 2026-11-08
- **核对范围：** 已通读官方 Introduction、Installation、Processors CLI Arguments 三份技术页；浏览 facefusion.io 产品主页时抓取结果无可解析正文，因此不对主页营销和授权细节作额外推断。
- **通读：** 已完成
- **开源状态：** GitHub 官方组织公开应用源码；本归档不等同于对所有预训练模型、商用部署或第三方权重的许可认证。

## 一手官方文档

1. [Introduction — 产品定位和可用 CLI 命令](https://docs.facefusion.io/)
2. [Installation — Python 3.12 Conda 环境、后端安装及启动](https://docs.facefusion.io/installation)
3. [Processors — 多处理器组合及可选模块](https://docs.facefusion.io/usage/cli-arguments/processors)
4. [GitHub README](https://github.com/facefusion/facefusion/blob/master/README.md)

## 已核对结论

- 官方定位是 face manipulation platform，默认处理器为 `face_swapper`，可用 `--processors face_swapper face_enhancer` 组合任务。
- 官方安装文档给出的启动示例：`python facefusion.py run --open-browser`；也提供 `headless-run`、`batch-run` 和 job 管理相关命令。
- 配套文档列举 `age_modifier`、`expression_restorer`、`face_debugger`、`face_editor`、`face_enhancer`、`face_swapper`、`frame_colorizer`、`frame_enhancer`、`lip_syncer`；实际动态发现和完整可用性以对应 commit 的代码与运行环境为准。
- 文档的 CUDA 安装示例指定 `python install.py cuda@12`；不可由此推断 CUDA 13 或其他驱动组合无须适配。

## 对 wiki 的映射

- [FaceFusion 项目实体](../../wiki/entities/facefusion.md)
- [官方仓库源码深读归档](../repos/facefusion.md)
