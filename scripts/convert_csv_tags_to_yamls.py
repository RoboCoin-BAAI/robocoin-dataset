import draccus

from robocoin_dataset.csv_tags_to_yaml_util import (
  ConvertCsvTags2YamlsConfig,
  ConvertCsvTags2YamlsUtil,
)


def main() -> None:
  """
  Main entry point for the CSV tags to YAML converter.

  Parses command line configuration and runs the conversion process.
  """
  config = draccus.parse(ConvertCsvTags2YamlsConfig)
  convetor = ConvertCsvTags2YamlsUtil(config)
  convetor.generate_dataset_tags_files()


if __name__ == "__main__":
  main()
