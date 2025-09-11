import json
import logging
import shutil
from pathlib import Path

import numpy as np
import pandas as pd

from robocoin_dataset.format_converter.tolerobot.lerobot_format_converter import (
    LerobotFormatConverter,
)


class LerobotFormatConverterLerobot(LerobotFormatConverter):
    """
    Converter for LeRobot format datasets to our standard format.
    Handles:
    1. Reading parquet data from LeRobot datasets
    2. Extracting and converting various data types (states, actions, images)
    3. Restructuring data according to our configuration mapping
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
        # Store dataset-specific paths BEFORE calling parent constructor
        self.lerobot_data_path = Path(dataset_path)
        self.parquet_files = list((self.lerobot_data_path / "data" / "chunk-000").glob("*.parquet"))
        self.videos_path = self.lerobot_data_path / "videos"
        
        # Load dataset metadata
        self._load_metadata()
        
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
            self.logger.info(f"Loaded {len(self.episodes)} episodes and {len(self.tasks_metadata)} tasks from LeRobot dataset")

    def _prevalidate_files(self) -> None:
        """Validate that the LeRobot dataset has required structure and files"""
        dataset_path = Path(self.dataset_path)
        
        # Check if dataset directory exists
        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset path {dataset_path} does not exist")
        
        # Check for required directories
        required_dirs = ["data", "meta", "videos"]
        missing_dirs = []
        for dir_name in required_dirs:
            dir_path = dataset_path / dir_name
            if not dir_path.exists():
                missing_dirs.append(str(dir_path))
        
        if missing_dirs:
            if self.logger:
                self.logger.warning(f"Missing directories in LeRobot dataset: {missing_dirs}")
        
        # Check for data files
        data_dir = dataset_path / "data" / "chunk-000"
        if data_dir.exists():
            parquet_files = list(data_dir.glob("*.parquet"))
            if not parquet_files:
                if self.logger:
                    self.logger.warning(f"No parquet files found in {data_dir}")
        else:
            if self.logger:
                self.logger.warning(f"Data directory {data_dir} does not exist")
        
        # Check for metadata files
        meta_dir = dataset_path / "meta"
        if meta_dir.exists():
            required_meta_files = ["episodes.jsonl", "tasks.jsonl"]
            for meta_file in required_meta_files:
                meta_file_path = meta_dir / meta_file
                if not meta_file_path.exists():
                    if self.logger:
                        self.logger.warning(f"Missing metadata file: {meta_file_path}")
        
        # Check for videos directory
        videos_dir = dataset_path / "videos"
        if videos_dir.exists() and self.logger:
            video_chunks = list(videos_dir.glob("chunk-*"))
            self.logger.info(f"Found {len(video_chunks)} video chunks in {videos_dir}")

    def _load_metadata(self) -> None:
        """Load LeRobot dataset metadata"""
        
        # Load episodes info
        episodes_file = self.lerobot_data_path / "meta" / "episodes.jsonl"
        self.episodes = []
        if episodes_file.exists():
            with open(episodes_file) as f:
                for line in f:
                    self.episodes.append(json.loads(line.strip()))
        
        # Load tasks info
        tasks_file = self.lerobot_data_path / "meta" / "tasks.jsonl"
        self.tasks_metadata = []
        if tasks_file.exists():
            with open(tasks_file) as f:
                for line in f:
                    self.tasks_metadata.append(json.loads(line.strip()))
        
        # Log after parent constructor is called (when logger is available)
        # This will be called again after super().__init__ if needed

    def _get_tasks(self) -> list[str]:
        """Override to provide generic task structure for LeRobot datasets"""
        # For generic LeRobot datasets, we can derive task names from metadata
        # or default to a single manipulation task
        if self.tasks_metadata:
            return [task.get("task_name", "manipulation_task") for task in self.tasks_metadata]
        return ["manipulation_task"]

    def _get_dataset_task_paths(self) -> dict[Path, str]:
        """Override to provide generic task path mapping for LeRobot datasets"""
        tasks = self._get_tasks()
        # For LeRobot format, all data is typically in one location
        return {self.lerobot_data_path: tasks[0]}

    def _get_task_episodes_num(self, task_path: Path) -> int:
        """Override to provide episode count from parquet files"""
        return len(self.parquet_files)









    def _move_videos_to_standard_structure(self) -> None:
        """Move videos to our standard structure if needed"""
        # Handle depth videos if they exist in a separate depth/ folder
        depth_path = self.lerobot_data_path / "depth"
        if depth_path.exists():
            target_videos_path = Path(self.output_path) / "videos"
            target_videos_path.mkdir(parents=True, exist_ok=True)
            
            # Copy depth videos structure to videos folder
            for chunk_dir in depth_path.iterdir():
                if chunk_dir.is_dir():
                    target_chunk_dir = target_videos_path / chunk_dir.name
                    target_chunk_dir.mkdir(exist_ok=True)
                    
                    for camera_dir in chunk_dir.iterdir():
                        if camera_dir.is_dir():
                            # Keep original camera naming convention
                            target_camera_dir = target_chunk_dir / camera_dir.name
                            
                            if target_camera_dir.exists():
                                shutil.rmtree(target_camera_dir)
                            shutil.copytree(camera_dir, target_camera_dir)
                            
                            if self.logger:
                                self.logger.info(f"Moved videos: {camera_dir} -> {target_camera_dir}")

    def _convert_units(self, data: np.ndarray, data_type: str) -> np.ndarray:
        """
        Convert data units to our standard if needed:
        - All angles in radians
        - All distances in meters
        """
        converted_data = data.copy().astype(np.float32)
        
        # Generic unit conversion logic
        # Specific conversions can be added based on data_type
        if data_type in ["joint_position", "joint_effort", "joint_velocity"]:
            # Assume joint data is already in standard units (radians, etc.)
            pass
        elif data_type in ["end_position", "translation"]:
            # Assume position data is already in meters
            pass
        elif data_type == "orientation":
            # Assume quaternions are already normalized
            pass
        
        return converted_data

    def _load_episode_data(self, ep_idx: int) -> pd.DataFrame:
        """Load parquet data for a specific episode"""
        if ep_idx >= len(self.parquet_files):
            raise ValueError(f"Episode {ep_idx} not found. Available: {len(self.parquet_files)}")
        
        parquet_file = self.parquet_files[ep_idx]
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
        For LeRobot format conversion, we typically don't process individual frame images
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
        return self._convert_units(data, "state")

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
        return self._convert_units(data, "action")

    def _get_episode_frames_num(self, task_path: Path, ep_idx: int) -> int:
        """Get number of frames in an episode"""
        df = self._load_episode_data(ep_idx)
        return len(df)

    def _get_task_episodes_num(self, task_path: Path) -> int:
        """Get number of episodes in a task"""
        return len(self.parquet_files)

    def _prepare_episode_states_buffer(self, task_path: Path, ep_idx: int) -> pd.DataFrame:
        """Prepare states buffer by loading the episode data"""
        return self._load_episode_data(ep_idx)

    def _prepare_episode_actions_buffer(self, task_path: Path, ep_idx: int) -> pd.DataFrame:
        """Prepare actions buffer by loading the episode data"""
        return self._load_episode_data(ep_idx)



    def _get_episode_task(self, ep_idx: int) -> str:
        """Get task name for an episode"""
        if ep_idx < len(self.episodes):
            # Try to get task from episode metadata
            episode_info = self.episodes[ep_idx]
            return episode_info.get("task", f"episode_{ep_idx:06d}")
        
        return f"episode_{ep_idx:06d}"

