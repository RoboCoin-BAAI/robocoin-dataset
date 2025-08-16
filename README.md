# RoboCoin Dataset 文档

欢迎使用 `robocoin-dataset` —— 一个用于构建 robocoin 数据集的 Python 工具集项目。

本项目采用 [`uv`](https://github.com/astral-sh/uv) 作为默认的包管理工具。

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

### 1. 创建conda环境克隆并安装必要库

```bash
conda create -n robocoin-dataset
conda activate robocoin-dataset
conda install ffmpeg=7.1.1 -c conda-forge
``` 

### 2. 克隆并进入项目目录

```bash
git clone --recursive https://github.com/RoboCoin-BAAI/robocoin-dataset.git
cd robocoin-dataset
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
uv pip install thirdparty/robocoin-lerobot
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
```bash
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
