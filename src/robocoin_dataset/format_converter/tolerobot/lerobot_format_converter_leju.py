import json
import logging
import shutil
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import pandas as pd

from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)


class LerobotFormatConverterLeju(LerobotFormatConverter):
    """
    Converter for Leju robot data from one LeRobot format to our standard format.
    Handles:
    1. Moving depth videos from depth/ to videos/ folder
    2. Converting units (angles to radians, distances to meters)
    3. Restructuring parquet data according to our configuration
    """

    def __init__(
        self,
        dataset_path: str,
        output_path: str,
        converter_config: dict,
        repo_id: str,
        device_model: str | None = None,
        logger: logging.Logger | None = None,
        video_backend: str = "pyav",
        image_writer_processes: int = 4,
        image_writer_threads: int = 4,
    ) -> None:
        # Store Leju-specific paths BEFORE calling parent constructor
        self.leju_data_path = Path(dataset_path)
        self.leju_parquet_files = list((self.leju_data_path / "data" / "chunk-000").glob("*.parquet"))
        self.leju_depth_path = self.leju_data_path / "depth"
        self.leju_videos_path = self.leju_data_path / "videos"
        
        # Load Leju metadata
        self._load_leju_metadata()
        
        # Call parent constructor
        super().__init__(
            dataset_path=dataset_path,
            output_path=output_path,
            converter_config=converter_config,
            repo_id=repo_id,
            device_model=device_model,
            logger=logger,
            video_backend=video_backend,
            image_writer_processes=image_writer_processes,
            image_writer_threads=image_writer_threads,
        )
        
        # Log metadata info now that logger is available
        if self.logger:
            self.logger.info(f"Loaded {len(self.leju_episodes)} episodes and {len(self.leju_tasks)} tasks from Leju dataset")

    def _load_leju_metadata(self) -> None:
        """Load Leju dataset metadata"""
        
        # Load episodes info
        episodes_file = self.leju_data_path / "meta" / "episodes.jsonl"
        self.leju_episodes = []
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    self.leju_episodes.append(json.loads(line.strip()))
        
        # Load tasks info
        tasks_file = self.leju_data_path / "meta" / "tasks.jsonl"
        self.leju_tasks = []
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    self.leju_tasks.append(json.loads(line.strip()))
        
        # Log after parent constructor is called (when logger is available)
        # This will be called again after super().__init__ if needed

    def _get_leju_episodes_count(self) -> int:
        """Get the number of episodes in Leju dataset"""
        if hasattr(self, 'leju_episodes') and self.leju_episodes:
            return len(self.leju_episodes)
        
        # Fallback: count parquet files or episode directories
        if self.leju_parquet_files:
            return len(self.leju_parquet_files)
        
        return 0

    def _prepare_leju_episode_buffers(self, ep_idx: int) -> tuple[any, any, any]:
        """Prepare buffers for a Leju episode"""
        # For Leju, we can load the episode data as buffers
        episode_data = self._load_episode_data(ep_idx)
        return episode_data, episode_data, episode_data  # images, states, actions from same source

    def _get_leju_episode_frames_num(self, ep_idx: int) -> int:
        """Get the number of frames in a Leju episode"""
        try:
            episode_data = self._load_episode_data(ep_idx)
            if episode_data is not None and hasattr(episode_data, '__len__'):
                return len(episode_data)
            return 0
        except Exception:
            return 0

    def _get_leju_frame_data(
        self,
        ep_idx: int,
        frame_idx: int,
        images_buffer: any = None,
        states_buffer: any = None,
        actions_buffer: any = None,
    ) -> dict[str, np.ndarray]:
        """Get frame data for Leju format"""
        # Use the existing _get_lerobot_datas method with Leju-specific task path
        # For Leju, we'll use the main data path as task_path
        task_path = self.leju_data_path
        
        return self._get_lerobot_datas(
            task_path=task_path,
            ep_idx=ep_idx,
            frame_idx=frame_idx,
            images_buffer=images_buffer,
            states_buffer=states_buffer,
            actions_buffer=actions_buffer,
        )

    def _move_depth_videos_to_videos(self) -> None:
        """Move depth videos from depth/ folder to videos/ folder"""
        if not self.leju_depth_path.exists():
            if self.logger:
                self.logger.warning("No depth folder found in Leju dataset")
            return
        
        target_videos_path = Path(self.output_path) / "videos"
        target_videos_path.mkdir(parents=True, exist_ok=True)
        
        # Copy depth videos structure to videos folder
        for chunk_dir in self.leju_depth_path.iterdir():
            if chunk_dir.is_dir():
                target_chunk_dir = target_videos_path / chunk_dir.name
                target_chunk_dir.mkdir(exist_ok=True)
                
                for camera_dir in chunk_dir.iterdir():
                    if camera_dir.is_dir():
                        # Convert depth camera naming to our convention
                        camera_name = camera_dir.name.replace("observation.videos.depth.", "observation.videos.depth.")
                        target_camera_dir = target_chunk_dir / camera_name
                        
                        if target_camera_dir.exists():
                            shutil.rmtree(target_camera_dir)
                        shutil.copytree(camera_dir, target_camera_dir)
                        
                        if self.logger:
                            self.logger.info(f"Moved depth videos: {camera_dir} -> {target_camera_dir}")

    def _convert_leju_units(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Convert Leju data units to our standard:
        - All angles in radians
        - All distances in meters
        """
        converted_data = data.copy()
        
        # Based on the data analysis, Leju data seems to already be in radians
        # since the range is [-1.681, 1.000] which is reasonable for joint angles in radians
        # If needed, we can add specific conversions here
        
        if data_type in ["joint_position", "joint_effort", "joint_velocity"]:
            # Joint data appears to already be in radians
            pass
        elif data_type in ["end_position", "translation"]:
            # Position data - check if conversion needed
            # Current range suggests it might already be in meters
            pass
        elif data_type == "orientation":
            # Quaternions - already normalized
            pass
        
        return converted_data.astype(np.float32)

    def _load_episode_data(self, ep_idx: int) -> pd.DataFrame:
        """Load parquet data for a specific episode"""
        if ep_idx >= len(self.leju_parquet_files):
            raise ValueError(f"Episode {ep_idx} not found. Available: {len(self.leju_parquet_files)}")
        
        parquet_file = self.leju_parquet_files[ep_idx]
        df = pd.read_parquet(parquet_file)
        
        if self.logger:
            self.logger.debug(f"Loaded episode {ep_idx} with {len(df)} frames from {parquet_file}")
        
        return df

    # Abstract method implementations for base class compatibility
    def _get_frame_image(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        images_buffer: any = None,
    ) -> np.ndarray:
        """
        For Leju conversion, we don't process individual frame images
        since videos are handled by moving files
        """
        # Return dummy image data - this won't be used in our conversion process
        return np.zeros((480, 848, 3), dtype=np.uint8)

    def _get_frame_sub_states(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_states_buffer: any = None,
    ) -> np.ndarray:
        """Extract state data for a specific frame"""
        df = sub_states_buffer if sub_states_buffer is not None else self._load_episode_data(ep_idx)
        
        if frame_idx >= len(df):
            raise ValueError(f"Frame {frame_idx} not found in episode {ep_idx}")
        
        # Extract the requested state data based on args_dict
        field_name = args_dict.get("field_name", "observation.state")
        
        if field_name not in df.columns:
            raise ValueError(f"Field {field_name} not found in data")
        
        frame_data = df.iloc[frame_idx][field_name]
        
        # Convert to numpy array and apply unit conversion
        data = np.array(frame_data, dtype=np.float32)
        return self._convert_leju_units(data, "state")

    def _get_frame_sub_actions(
        self,
        task_path: Path,
        ep_idx: int,
        frame_idx: int,
        args_dict: dict,
        sub_actions_buffer: any = None,
    ) -> np.ndarray:
        """Extract action data for a specific frame"""
        df = sub_actions_buffer if sub_actions_buffer is not None else self._load_episode_data(ep_idx)
        
        if frame_idx >= len(df):
            raise ValueError(f"Frame {frame_idx} not found in episode {ep_idx}")
        
        # Extract the requested action data based on args_dict
        field_name = args_dict.get("field_name", "action")
        
        if field_name not in df.columns:
            raise ValueError(f"Field {field_name} not found in data")
        
        frame_data = df.iloc[frame_idx][field_name]
        
        # Convert to numpy array and apply unit conversion
        data = np.array(frame_data, dtype=np.float32)
        return self._convert_leju_units(data, "action")

    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        """Get number of frames in an episode"""
        df = self._load_episode_data(ep_idx)
        return len(df)

    def _get_task_episodes_num(self, task_path: Path) -> int:
        """Get number of episodes in a task"""
        return len(self.leju_parquet_files)

    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> pd.DataFrame:
        """Prepare states buffer by loading the episode data"""
        return self._load_episode_data(ep_idx)

    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> pd.DataFrame:
        """Prepare actions buffer by loading the episode data"""
        return self._load_episode_data(ep_idx)

    def convert_leju(self) -> Iterable[tuple[str, int, int]]:
        """
        Convert Leju dataset to our standard format.
        
        This method:
        1. Moves depth videos to videos folder
        2. Converts parquet data with unit conversion
        3. Yields progress information
        """
        if self.logger:
            self.logger.info("Starting Leju dataset conversion")
        
        # Step 1: Use the parent class convert_leju method for Leju-specific conversion
        # This will handle dataset creation first
        yield from super().convert_leju()
        
        # Step 2: After conversion, move depth videos to the created dataset
        self._move_depth_videos_to_videos()
        
        if self.logger:
            self.logger.info("Leju dataset conversion completed")

    def _get_episode_task(self, ep_idx: int) -> str:
        """Get task name for an episode"""
        if ep_idx < len(self.leju_episodes):
            # Try to get task from episode metadata
            episode_info = self.leju_episodes[ep_idx]
            return episode_info.get("task", f"episode_{ep_idx:06d}")
        
        return f"episode_{ep_idx:06d}"
