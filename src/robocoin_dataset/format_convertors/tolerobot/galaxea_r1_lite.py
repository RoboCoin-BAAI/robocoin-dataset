from robocoin_dataset.format_convertors.tolerobot.lerobot_format_convertor import (
    LerobotFormatConvertor,
)


class GalaxeaR1Lite(LerobotFormatConvertor):
    def __init__(self, dataset_path: str, lerobot_dst_path: str) -> None:
        super().__init__(dataset_path, lerobot_dst_path)
