# RIPPLE / Wave Rider 官方项目页

- **类型：** site（作者交互演示与项目资料）
- **链接：** [项目主页](https://fly-pigth.github.io/wave_rider/) · [论文 HTML](https://arxiv.org/html/2611.34569v1)
- **核对日期：** 2026-11-27
- **通读：** 已完成
- **核对范围：** 主页、交互说明、[浏览器入口 JS](https://fly-pigth.github.io/wave_rider/assets/index-Ct2y_tpt.js)、[资源清单](https://fly-pigth.github.io/wave_rider/manifest.json) 和 [ElasticMembrane 场景](https://fly-pigth.github.io/wave_rider/scenes/springnet.xml)；提取部署 JS 的推理类，对两种公开演示策略做 CPU 合成输入检查。没有运行完整浏览器物理或真机。
- **开源边界：** 网站发布运行后的 JS、MJCF 和用于网页的策略参数；Code 按钮仍禁用并标为 Released soon。未查到论文对应的 MJLab/PPO 训练仓或 MagicBot Z1 真机部署包，不从论文结论措辞推定已公开或把网页资产视为可再分发的正式模型发布。
- **沉淀到 wiki：** [RIPPLE](../../wiki/entities/paper-ripple-elastic-terrain.md)

## 可核查的浏览器资产

- 清单给出 springnet 与 jump 的 312 维输入、23 维输出，以及场景的 2 ms 步长和 10 次 decimation。可交互的 Policies 走实际的 MuJoCo WASM 和 JS 推理；部分 Applications 则使用预录轨迹，不能混为实时策略。
- 浏览器网络类在 CPU 上用两个站点提供的参数文件均得到有限输出，并拒绝 311 维输入。两文件下载字节的 SHA-256 与同日 manifest 各自的 `sha256` 字段未一致；只确认接口与当次响应，不确认与论文检查点或真机权重同一。未在这里转存权重。
