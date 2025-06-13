# Template Project 文档

欢迎使用 `template-project` —— 一个标准化的 Python 项目模板。

本项目旨在为你提供一个可复用、结构清晰、易于维护的 Python 项目基础结构，适合用于开发库、命令行工具或服务类应用，并采用 [`uv`](https://github.com/astral-sh/uv) 作为默认的包管理工具。

---

## 🛠 使用 uv 管理项目

[`uv`](https://github.com/astral-sh/uv) 是一个超快的 Python 包安装器和虚拟环境管理器，兼容 `pyproject.toml` 和 `pip` 生态，是替代 `pip`, `poetry`, `pipenv` 的现代化选择。

### 安装 uv（如尚未安装）

```bash
pip install uv
```

---

## 🧪 标准开发流程（推荐）

以下是开发者常见的工作流程：

### 1. 克隆并进入项目目录

```bash
git clone https://github.com/yourname/template-project.git
cd template-project
```

### 2. 创建虚拟环境

```bash
uv venv
```

这将在项目根目录下创建 `.venv` 虚拟环境。

### 3. 激活虚拟环境

```bash
source .venv/bin/activate    # Linux/macOS
# 或：
.venv\Scripts\activate       # Windows (PowerShell)
```

> 💡 提示：如果你不激活环境，`uv` 默认也会使用 `.venv`。

### 4. 安装项目依赖

```bash
uv sync
```

#### 编辑模式

```bash
uv pip install -e .
```
#### 开发模式

```bash
uv pip install -e .[dev]
```

### 5. 安装 pre-commit 钩子（提交前自动格式化和检查代码）
```
pre-commit install
```

---

## 🔧 常用依赖管理操作

### 添加新依赖

推荐在`pyproject.toml`文件中手动添加依赖项，或使用以下命令。
```bash
uv add requests           # 添加运行时依赖
uv add --test ruff       # 添加开发依赖
```

这些命令会自动更新 `pyproject.toml` 和 `uv.lock` 文件。

### 添加或删除依赖

在pyproject.toml中的相应位置，添加或删除依赖项

### 查看已安装依赖

```bash
uv pip list
```

---

## 🧹 清理与重建环境（可选）

```bash
# 删除虚拟环境
rm -rf .venv/

# 重新创建和安装
uv venv
uv pip install -e .
```

---

## 🚀 快速开始开发

```bash
# 运行主程序
template_project

# 或运行测试
uv pip install -e .[test]
pytest tests/
```

---

## 📁 项目结构概览

```text
.
├── examples/             # 使用示例代码
├── src/     # 主模块代码
├── tests/                # 单元测试
├── docs/                 # 文档目录
├── pyproject.toml        # 项目配置与依赖管理
├── uv.lock               # 依赖锁定文件
├── .gitignore            # Git 忽略规则
├── README.md             # 项目简介
└── CONTRIBUTING.md       # 贡献指南
```

---

## 🛠 开发者指南

- 想要贡献代码？请参考 [CONTRIBUTION.md](../CONTRIBUTION.md)
- 想了解如何编写测试？请查看 `tests/` 目录
- 想了解如何格式化代码？请使用 `ruff format .` 和 `ruff check .`

---

## 📄 许可证

本项目采用 [MIT License](../LICENSE) 开源协议。

---

感谢你的使用与支持！如果你有任何问题或建议，请在 GitHub 上提交 issue。
