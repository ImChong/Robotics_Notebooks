# 自托管字体

## DM Mono

- 用途：站点标题 / 规模数字的 `--font-display`（Technical Blueprint 工程感等宽体）。
- 版本：Google Fonts `dmmono` v16，仅取 `latin` 与 `latin-ext` 子集，字重 400 / 500。
- 来源：`https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&display=swap` 解析出的 `fonts.gstatic.com` woff2。
- 许可：SIL Open Font License 1.1，<https://openfontlicense.org/>。
- 自托管而非引 Google Fonts CDN 的原因：不新增第三方源；`sw.js` 只缓存同源请求，同源托管才能进离线缓存。

| 文件 | 字重 | 子集 |
|------|------|------|
| `dm-mono-400-latin.woff2` | 400 | latin |
| `dm-mono-400-latin-ext.woff2` | 400 | latin-ext |
| `dm-mono-500-latin.woff2` | 500 | latin |
| `dm-mono-500-latin-ext.woff2` | 500 | latin-ext |

`docs/style.css` 的 `@font-face` 保留了各子集的 `unicode-range`，CJK 字符不会命中 DM Mono，会按 `--font-sans` 正常渲染。
