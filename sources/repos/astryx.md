# Astryx（Meta 开源设计系统）— 官方仓库原文与源码核验

- **标题：** Astryx — An open source design system that's fully customizable and agent ready
- **类型：** repo / 官方技术文档 / 核心源码
- **维护机构：** Meta
- **链接：** https://github.com/meta/astryx
- **源码快照：** 143a5de8f5314a9af56b0dc15122ba9fdc0a5a91（2026-11-07 核查 main HEAD）
- **官方站点：** https://astryx.atmeta.com/；[站点归档](../sites/astryx.md)
- **入库日期：** 2026-11-07
- **许可证：** MIT（上游仓库 README 与 package.json 标识；许可证适用范围以对应 LICENSE 原文为准）
- **开源状态：** 已开源（组件与 CLI 源码公开；不涉及机器人训练模型或数据集）
- **核对范围：** 完整阅读上游 README.md、packages/core/README.md、packages/cli/README.md；检查 packages/core/package.json、packages/core/src/Button/{Button.tsx,index.ts}、packages/core/src/theme/Theme.tsx、packages/cli/clients/cli/{bin/astryx.mjs,run.mjs,index.mjs,commands/init.mjs}、packages/cli/api/init/{init.mjs,run/run.mjs} 的入口、依赖与执行路径。源码审读限定上述实现，不代表逐文件审查 180+ 组件。
- **通读：** 已完成

## 核验结论

1. Astryx 是 React 19+ 的 UI 设计系统，采用 StyleX 编写组件样式，消费者可使用预编译 CSS，无须为普通用法安装样式构建插件；CLI 另为开发辅助。
2. 根 README 介绍 150+ 组件、7 套内置主题、dark mode、模板、CLI；官方首页在核查日写「Over 180 components」。不同页面口径不一致，均按来源描述，不统一成一个未经证实的精确数。
3. 安装依赖：@astryxdesign/core、@astryxdesign/theme-neutral、@stylexjs/stylex；CLI 对应 @astryxdesign/cli。包的 peer dependencies 包括 React/ReactDOM 19+。
4. packages/core/README.md 的 Next.js 示例按 reset.css → astryx.css → theme.css 顺序导入，再用 Theme 包装应用。CSS 层分别是 reset、astryx-base、astryx-theme；Tailwind 可选。
5. Theme.tsx 中主题令牌作为 CSS 自定义属性生效；根层 Theme 将 data-theme 和 data-astryx-theme 同步到 document.documentElement，内嵌 Theme 不执行根同步。已构建主题不走运行时 CSS 注入分支。
6. CLI 的 bin/astryx.mjs 首先检查 Node.js 支持版本，然后由 run.mjs 协调版本选择、构建命令与解析；init 命令由 commands/init.mjs 调 api/init/init.mjs，默认再分派给 api/init/run/run.mjs，写入 Agent 文档提示并返回结果。
7. CLI README 明确 npx 未安装时须使用 scoped 形式 npx @astryxdesign/cli；裸用 npx astryx 会匹配无关包。CLI 可检索组件、模板、文档，且支持 swizzle、主题构建与 codemod。
8. 源码核查了 Theme 与 Button 的入口、样式及上下文，但没有本地安装、浏览器可访问性实测、构建性能评测，也没有核查每一个组件的内部实现。

## 源码入口与材料（固定版本）

- [根 README](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/README.md)
- [Core 说明](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/core/README.md)
- [CLI 说明](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/cli/README.md)
- [Theme.tsx](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/core/src/theme/Theme.tsx)
- [Button.tsx](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/core/src/Button/Button.tsx)
- [CLI bin](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/cli/clients/cli/bin/astryx.mjs)
- [CLI run](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/cli/clients/cli/run.mjs)
- [init 命令](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/cli/clients/cli/commands/init.mjs)
- [init API](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/cli/api/init/init.mjs)
- [init run](https://github.com/meta/astryx/blob/143a5de8f5314a9af56b0dc15122ba9fdc0a5a91/packages/cli/api/init/run/run.mjs)

## 对 wiki 的映射

- [Astryx：面向人与 Agent 的可定制 React 设计系统](../../wiki/entities/astryx.md)
