# print3d v0.1.0 | 官方版本与源码归档

- **类型：** repo / official software release
- **作者：** mfranzon
- **版本：** [v0.1.0](https://github.com/mfranzon/print3d/releases/tag/v0.1.0) · [固定提交 49f8213](https://github.com/mfranzon/print3d/tree/49f821366e955f3a11de72f05d39f93700c7e30c)
- **许可：** MIT（仓库代码）；资料与照片单独见上游署名。
- **核对范围：** 发布说明、标签 README、工具工作流说明、`cli.py`、`mcp_server.py`、`pipeline.py`、切片配置来源、命令验证及自带测试样件。
- **通读：** 已完成
- **入库日期：** 2026-11-25
- **对应节点：** [print3d](../../wiki/entities/print3d.md)

v0.1.0 面向 Bambu Lab X2D 及 0.4 mm 喷嘴。描述、参考照片、商品尺寸可作为建模依据；照片尺标、报价原句及未知项纳入 `spec.json`，不代表自动恢复完整几何。建模来自操作者或 Agent 撰写的 Blender 脚本，不是工具本身自带的生成模型。

`cli.py::cmd_make` 和 MCP `make` 共用 `pipeline.py::run_job`。单件路径先运行建模脚本或读现有 STL，再做网格可制造性、可选轮廓对照，然后由 `studio.py::run_slice` 调用本地 Bambu Studio，读回已切片 3MF、检查指标、生成报告与预览。严重的初步网格错误可中断，其他问题也可能在切片后归入报告；没有“所有检查先过才切片”的单一总门禁。Hermes 插件声明服务为不受信任，写文件或运行脚本的调用应受审批。

标签固定配置来自 Bambu Studio 02.08.02.61、X2D 0.4 mm、0.20 mm Standard、PLA Basic 与纹理 PEI；不会替用户下发给打印机。CLI 已对标签自带样件验证 `check` 和 `inspect` 的读入与报告路径；未运行真实切片或打印。
