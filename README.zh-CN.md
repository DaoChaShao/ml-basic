<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**应用简介**
---
本仓库用于学习和理解神经网络的基础知识和原理。


**隐私声明**
---
本应用可能需要您输入个人信息或隐私数据，以生成定制建议和结果。但请放心，应用程序 **不会**
收集、存储或传输您的任何个人信息。所有计算和数据处理均在本地浏览器或运行环境中完成，**不会** 向任何外部服务器或第三方服务发送数据。

整个代码库是开放透明的，您可以随时查看 [这里](./) 的代码，以验证您的数据处理方式。

**许可协议**
---
本应用基于 **BSD-3-Clause 许可证** 开源发布。您可以点击链接阅读完整协议内容：👉 [BSD-3-Clause Licence](./LICENCE)。

**更新日志**
---
本指南概述了如何使用 git-changelog 自动生成并维护项目的变更日志的步骤。

1. 使用命令安装所需依赖：
    ```bash
    pip install git-changelog
    ```
2. 执行命令
    ```bahs
    pip show git-changelog
    ```
   或者
    ```bash
    pip show git-changelog | grep Version
    ```
   检查是否已安装该包及其版本。
3. 在项目根目录下准备配置文件 `pyproject.toml`。
4. 变更日志风格采用 [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/)。
5. 执行命令
    ```bash
    git-changelog --output CHANGELOG.md
    ```
   提交更改并更新变更日志。
6. 使用命令
    ```bash
    git push origin main
    ```
   将更改推送到远程仓库。
