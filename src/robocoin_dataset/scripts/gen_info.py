"""
RoboCoin Datasets Generate Dataset Readme.md from readme_template/readme.j2 template
usage:
python -m robocoin.datasets.gen_readme --config configs/upload.yaml
"""

import draccus

from robocoin_dataset.dataset_info_util import LocalDsInfoConfig, LocalDsInfoUtil

if __name__ == "__main__":
  """
    Main entry point for the dataset info generator.
    
    Parses command line configuration and runs the info generation process.
    """
  config = draccus.parse(LocalDsInfoConfig)
  generator = LocalDsInfoUtil(config)
  generator.generate_infos()
  pass
