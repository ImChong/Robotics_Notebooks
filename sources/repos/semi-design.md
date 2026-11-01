# Semi Design 官方仓库

- **类型：** React 组件库与设计系统
- **代码：** <https://github.com/DouyinFE/semi-design>（公开，MIT）
- **官网：** <https://semi.design/>
- **核对版本：** [0580665](https://github.com/DouyinFE/semi-design/tree/0580665e20da61422b5269b0c5c44aa2b389d2dc)，2026-11-01 查阅。
- **核对范围：** README 全文、Button 组件入口和基础 Button 实现；设计工具和其他组件源码尚未逐项审阅。

README 给出 80+ 组件、3000+ Design Token、Figma 转码、国际化、RTL 和 SSR 支持，安装命令 `npm install @douyinfe/semi-ui`。示例通过 `createRoot` 渲染 Form 与 Button；`packages/semi-ui/button/index.tsx` 根据 icon、loading 和 disabled 选择 IconButton 或基础 Button。基础 Button 生成 HTML button，传递事件和无障碍属性。

- [官网归档](../sites/semi-design.md)
- [知识实体](../../wiki/entities/semi-design.md)
