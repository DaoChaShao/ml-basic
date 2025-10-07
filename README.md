<p align="right">
  Language Switch / 语言选择：
  <a href="./README.zh-CN.md">🇨🇳 中文</a> | <a href="./README.md">🇬🇧 English</a>
</p>

**INTRODUCTION**
---
This repository is used to understand the fundamentals and principles of Neural Networks.


**PRIVACY NOTICE**
---
This application may require inputting personal information or private data to generate customised suggestions,
recommendations, and necessary results. However, please rest assured that the application does **NOT** collect, store,
or transmit your personal information. All processing occurs locally in the browser or runtime environment, and **NO**
data is sent to any external server or third-party service. The entire codebase is open and transparent — you are
welcome to review the code [here](./) at any time to verify how your data is handled.

**LICENCE**
---
This application is licensed under the [BSD-3-Clause Licence](LICENCE). You can click the link to read the licence.

**CHANGELOG**
---
This guide outlines the steps to automatically generate and maintain a project changelog using git-changelog.

1. Install the required dependencies with the command:
    ```bash
    pip install git-changelog
    ```
2. Run the command
    ```bahs
    pip show git-changelog
    ```
   or
    ```bash
    pip show git-changelog | grep Version
    ```
   to check whether the changelog package has been installed and its version.
3. Prepare the configuration file of `pyproject.toml` at the root of the file.
4. The changelog style is [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/).
5. Run the command
    ```bash
    git-changelog --output CHANGELOG.md
    ```
   to commit the changes and updating the changelog.
6. Push the changes to the remote repository with the command
    ```bash
    git push origin main
    ```
   or using the UI interface button named `Git -> Push`.