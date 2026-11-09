# CamX 数据集网站与样本处理工具（官方仓库）

- **GitHub：** https://github.com/camx-anonymous/camx-anonymous.github.io
- **项目站点：** https://camx-anonymous.github.io/camx-dataset/
- **核查日期：** 2026-11-09
- **核对范围：** 仓库 README、首页/数据页 HTML、data/pending.json、tools/build_site_data.py 的入口和许可调用、tools/trim_noncommercial.py 的规则、tools/build_samples.py 与 tools/publish_samples.py 的入口；未运行数据重建和发行下载。
- **通读：** 未完成

## 一手源码位置和行为

| 路径 | 已核对用途 |
|---|---|
| camx-dataset/index.html | 数据树、数据源引用/许可、样本命令和相机视频对照 |
| data/datasets.json / data/catalog/ | 数据记录与拆分的静态站点索引 |
| data/pending.json | 2026-10-30 统计口径：17,102.6 h / 1,794,319 episodes；另注明未裁剪附录 25,850.7 h |
| data/sources.json | 来源级论文引用和许可，不等于数据集自身单一宽松许可 |
| tools/build_site_data.py | 合并数据目录与统计，调用 trim_pass 进行许可过滤，按构建分支附加样本 |
| tools/trim_noncommercial.py | 去除若干非商业来源；RH20T 只保留 C 场景 |
| tools/build_samples.py | 从已有 LeRobot 数据中抽取 episode 0，生成每数据集一包的样本归档 |
| tools/publish_samples.py | 将包上传至 samples-v1-* GitHub Releases |

## 复现入口（需先具备数据和环境）

```bash
python3 tools/build_site_data.py
python3 tools/build_samples.py --root /path/to/camx_480p --out /path/to/camx_samples
python3 tools/publish_samples.py
python3 tools/build_site_data.py --attach-samples-only
```

README 规定样本是一个完整 LeRobot-v3 结构（meta/、data/、videos/）但只含一个 episode；build_site_data 附着样本 URL 后，网页按许可和引用要求生成针对选中切片的下载命令。

## 开放性与局限

- 仓库公开的是**数据站、索引与数据处理/样本发布代码**，不是 CamUVA 模型训练代码。
- 原始数据树、部分渲染依赖与 GitHub Release 维护权限不随公开仓库一并授予；示例脚本无法单凭空仓重新生成完整数据。
- README 声明站点仅显示商业使用许可允许的来源，并主动剔除非商业数据；每个原来源许可证仍须单独核对。
- 此 GitHub 仓库没有根级 LICENSE 文件（按目录核查），不可将公开代码自动等同于某个明确 OSI 许可。

## 知识库映射

[CamX / CamUVA 主实体](../../wiki/entities/project-camx-camuva.md)；[官网归档](../sites/camx-anonymous-github-io.md)；[论文状态](../papers/camx-17k-open-source-robot-data.md)；[CamUVA 配套仓库](camuva-rollouts.md)。
