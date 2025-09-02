"""
RoboCoin Datasets Generate Dataset Readme.md from readme_template/readme.j2 template
usage:
python -m robocoin.datasets.gen_readme --config configs/upload.yaml
"""

import draccus

from robocoin_dataset.dataset_readme_util import LocalDsReadmeConfig, LocalDsReadmeUtil

if __name__ == "__main__":
  """
    Main entry point for the README generator.
    
    Parses command line configuration and runs the README generation process.
    """
  config = draccus.parse(LocalDsReadmeConfig)
  generator = LocalDsReadmeUtil(config)
  generator.generate_readmes()
  pass
