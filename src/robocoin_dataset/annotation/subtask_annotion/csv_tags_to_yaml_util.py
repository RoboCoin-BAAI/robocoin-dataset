import re
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

import yaml

from .local_datasets_util import LocalDsConfig, LocalDsUtil


@dataclass
class ConvertCsvTags2YamlsConfig(LocalDsConfig):
  """
  Configuration class for converting CSV tags to YAML files.

  Attributes:
      csv_file_path (str): Path to the input CSV file containing tags. Defaults to empty string.
      output_path (str): Path to the output directory for YAML files. Defaults to empty string.
  """

  csv_file_path: str = ""
  output_path: str = ""


class ConvertCsvTags2YamlsUtil(LocalDsUtil):
  """
  Utility class for converting CSV tag files to individual YAML configuration files.

  This class reads a CSV file containing dataset tags and generates separate YAML
  configuration files for each dataset, with support for hierarchical tag structures.

  Attributes:
      config (ConvertCsvTags2YamlsConfig): Configuration object for the converter.
      logger: Logger instance for the converter.
  """

  def __init__(self, config: ConvertCsvTags2YamlsConfig) -> None:
    """
    Initialize the converter with configuration.

    Args:
        config (ConvertCsvTags2YamlsConfig): Configuration object for the converter.
    """
    super().__init__(config)
    self.config = config

    self.logger = self.setup_logger(logger_name="CONVERT_CSV_TAGS_TO_YAMLS")

  @cached_property
  def output_path(self) -> Path:
    """
    Get the output path for YAML files with validation.

    Returns:
        Path: Absolute path to the output directory.

    Raises:
        ValueError: If output_path is not set in configuration.
    """
    if not self.config.output_path:
      raise ValueError("output_path is not set.")
    output_path = Path(self.config.output_path).expanduser().absolute()
    if not output_path.exists():
      output_path.mkdir(parents=True, exist_ok=True)
    return output_path

  @cached_property
  def csv_file_path(self) -> Path:
    """
    Get the CSV file path with validation.

    Returns:
        Path: Absolute path to the CSV file.

    Raises:
        ValueError: If CSV file path is not set in configuration.
        FileNotFoundError: If the CSV file does not exist.
    """
    if not self.config.csv_file_path:
      raise ValueError("CSV file path is not set.")
    csv_path = Path(self.config.csv_file_path).expanduser().absolute()
    if not csv_path.exists():
      raise FileNotFoundError(f"CSV file {csv_path} does not exist.")
    return csv_path

  def parse_hierarchy(self, path: str, delimiter: str = "-", max_levels: int = 5) -> dict:
    """
    Parse hierarchical path string, e.g. 'A-B-C' -> {'level1': 'A', 'level2': 'B', 'level3': 'C'}

    Args:
        path (str): Hierarchical path string to parse.
        delimiter (str): Delimiter used to separate hierarchy levels. Defaults to "-".
        max_levels (int): Maximum number of hierarchy levels to parse. Defaults to 5.

    Returns:
        dict: Dictionary with level keys and corresponding values.
    """
    if not path or not path.strip():
      return {"level1": None}
    parts = [p.strip() for p in path.split(delimiter) if p.strip()]
    return {f"level{i + 1}": parts[i] if i < len(parts) else None for i in range(max_levels)}

  def parse_annotation_with_hierarchy(
    self,
    text: str,
    hierarchy_keys: list[str] = None,
    delimiter: str = "-",
    auto_convert_numeric: bool = True,
  ) -> dict:
    """
    Enhanced tag parser with support for hierarchical fields.

    Args:
        text (str): Multi-line tag text, format like "category: A-B-C\nquantity: 3"
        hierarchy_keys (list[str]): Field names to be treated as hierarchical structures.
        delimiter (str): Hierarchy delimiter. Defaults to "-".
        auto_convert_numeric (bool): Whether to automatically convert numeric strings to int/float.

    Returns:
        dict: Parsed dictionary with structured data.
    """
    if hierarchy_keys is None:
      hierarchy_keys = ["物品类别", "物品", "category", "object_path", "物体类别", "分类"]

    def try_convert(value: str) -> int | float | str:
      """
      Attempt to convert string value to appropriate numeric type.

      Args:
          value (str): String value to convert.

      Returns:
          int | float | str: Converted value.
      """
      if not auto_convert_numeric:
        return value
      value = value.strip()
      if not value:
        return value
      # Integer
      if value.isdigit() or (value.startswith("-") and value[1:].isdigit()):
        return int(value)
      # Float
      try:
        return float(value)
      except ValueError:
        pass
      return value

    result = {}
    lines = [line.strip() for line in text.strip().splitlines() if line.strip()]
    pattern = re.compile(r"^([^:：]+?)\s*[:：]\s*(.+)$")

    for line in lines:
      match = pattern.match(line)
      if not match:
        continue

      key = match.group(1).strip()
      value_str = match.group(2).strip()

      # Split multiple values (supporting multiple separators)
      items = [item.strip() for item in re.split(r"[、，,\s;；]+", value_str) if item.strip()]

      if key in hierarchy_keys and items:
        first_item = items[0]
        hierarchy_data = self.parse_hierarchy(first_item, delimiter=delimiter)
        result[key] = hierarchy_data
      else:
        converted_items = [try_convert(item) for item in items]
        result[key] = converted_items[0] if len(converted_items) == 1 else converted_items

    return result

  def generate_dataset_tags_files(self) -> dict:
    """
    Read tags CSV file and generate a YAML configuration file for each dataset.

    Args:
        tags_file: Path to the CSV tags file
        output_dir: Directory for output YAML files. If None, uses {tags_file.parent}/ds_configs

    Returns:
        dict: Summary dictionary with ds_name as key and parsed structured tags as value
    """
    tags_file = self.csv_file_path
    output_dir = self.output_path

    tags_dict = {}

    import csv

    with open(tags_file, encoding="utf-8") as f:
      reader = csv.reader(f)
      headers = next(reader)

      try:
        address_nas_idx = headers.index("address_nas")
        tags_idx = headers.index("tags")
      except ValueError as e:
        self.logger.error(f"❌ CSV file missing required columns 'address_nas' or 'tags': {e}")
        return tags_dict

      for row in reader:
        if len(row) <= max(address_nas_idx, tags_idx):
          self.logger.warning(f"⚠️  Incomplete row data, skipping: {row}")
          continue

        address_nas = row[address_nas_idx].strip()
        if not address_nas:
          continue

        ds_name = Path(address_nas).name

        # Clean ds_name: keep only valid leading characters (letters, numbers, -_.)
        match = re.match(r"^[a-zA-Z0-9_\-\.]+", ds_name)
        ds_name = match.group(0) if match else "unknown_dataset"

        try:
          self.check_dataset_dir_valid(ds_name)
        except Exception as e:
          self.logger.info(f'❌ Dataset "{ds_name}" not found, skipping: {e}')
          continue

        raw_tags = row[tags_idx].strip()

        # Parse tag text
        if raw_tags:
          try:
            dataset_tag_dict = self.parse_annotation_with_hierarchy(raw_tags)
          except Exception as e:
            self.logger.error(f"❌ Failed to parse tags for {ds_name}: {e}")
            dataset_tag_dict = {"raw_tags": raw_tags, "parse_error": str(e)}
        else:
          dataset_tag_dict = {}

        # Store in main dictionary
        tags_dict[ds_name] = dataset_tag_dict

        # Generate YAML file
        yaml_path = output_dir / f"{ds_name}.yml"
        try:
          with open(yaml_path, "w+", encoding="utf-8") as yf:
            yaml.dump(
              dataset_tag_dict,
              yf,
              allow_unicode=True,
              sort_keys=False,
              indent=2,
              default_flow_style=False,
              width=1000,
            )
          self.logger.info(f"✅ Generated YAML: '{ds_name}.yml' -> {yaml_path}")
        except Exception as e:
          self.logger.error(f"❌ Failed to write YAML for {ds_name}: {e}")

    return tags_dict
