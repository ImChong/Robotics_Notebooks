# expo-ft（pd-perry/expo-ft）

> 来源归档（ingest）

- **标题：** EXPO-FT / Real-Time EXPO-FT
- **类型：** repo
- **链接：** <https://github.com/pd-perry/expo-ft>
- **项目页：** <https://pd-perry.github.io/expo-ft/> · Real-Time：<https://pd-perry.github.io/real-time-expo-ft/>
- **论文：** [2605.25477](https://arxiv.org/abs/2605.25477) · [2609.18207](https://arxiv.org/abs/2609.18207)
- **入库日期：** 2026-09-29
- **一句话说明：** CoRL 2026 **EXPO-FT** 与 **Real-Time EXPO-FT** 共用仓库：server（uv + 修改版 OpenPI）+ client（DROID 真机 actor）；需 clone `pd-perry/openpi` 与 `pd-perry/droid` 对应分支。
- **沉淀到 wiki：** [`paper-expo-ft`](../../wiki/entities/paper-expo-ft.md)、[`paper-real-time-expo-ft`](../../wiki/entities/paper-real-time-expo-ft.md)

## 开放程度

| 项 | 状态 |
|----|------|
| 代码 | **已开源** — `pd-perry/expo-ft`（2026-09-29 项目页 GitHub 按钮） |
| 依赖 fork | OpenPI（`expo_ft` 或 `real-time-expo-ft` 分支）、DROID 同分支 |
| 权重 / π0.5 | 按 README 与 OpenPI 文档配置（非单文件一键权重） |

## 复现入口（README）

- **EXPO-FT：** 静态/慢推理任务 → `uv sync` + clone openpi@`expo_ft` + droid
- **Real-Time EXPO-FT：** 控制步内放不下 forward → openpi/droid@`real-time-expo-ft`

## 对 wiki 的映射

- [expo_ft_arxiv_2605_25477.md](../papers/expo_ft_arxiv_2605_25477.md)
- [real_time_expo_ft_arxiv_2609_18207.md](../papers/real_time_expo_ft_arxiv_2609_18207.md)
- 基座算法：[expo.md](expo.md)
