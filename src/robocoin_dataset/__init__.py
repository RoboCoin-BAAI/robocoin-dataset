import os
from importlib.metadata import version

# Define the path to the README file relative to the package directory
readme_path = os.path.join(os.path.join(os.path.dirname(os.path.dirname(__file__)), "README.md"))

# Read the content of the README file
try:
  with open(readme_path, encoding="utf-8") as f:
    __doc__ = f.read()
except FileNotFoundError:
  __doc__ = "README file not found."

try:
  __version__ = version("robocoin-dataset")
except Exception:
  __version__ = "unknown"

# 导出功能
# from .dataset import load_dataset
# from .utils import helper_function

# __all__ = ["load_dataset", "helper_function", "__version__"]
