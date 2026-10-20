"""Dataset-specific handlers for VOS evaluation."""

import glob
import logging
import os
import subprocess
import sys
from typing import List

logger = logging.getLogger("ssl_eval.vos")


class DatasetHandler:
    """Base class for VOS dataset handling."""

    def __init__(self, data_path, subset, module_dir, video_list_file=None):
        self.data_path = data_path
        self.subset = subset
        self.module_dir = module_dir
        self.video_list_file = video_list_file

    def _read_video_list(self, default_list_path: str) -> List[str]:
        list_path = self.video_list_file or default_list_path
        with open(list_path, "r", encoding="utf-8") as handle:
            return [line.strip() for line in handle.readlines() if line.strip()]

    def get_video_list(self) -> List[str]:
        raise NotImplementedError

    def get_video_dir(self, video_name: str) -> str:
        raise NotImplementedError

    def get_first_seg_path(self, video_name: str, frame_list: List[str]) -> str:
        raise NotImplementedError

    def get_first_frame_output_name(self, frame_list: List[str]) -> str:
        raise NotImplementedError

    def run_evaluation(self, results_path: str) -> None:
        raise NotImplementedError


class DAVISHandler(DatasetHandler):
    """Handler for DAVIS 2017 dataset."""

    def get_video_list(self):
        list_path = os.path.join(self.data_path, "ImageSets", "2017", f"{self.subset}.txt")
        return self._read_video_list(list_path)

    def get_video_dir(self, video_name):
        return os.path.join(self.data_path, "JPEGImages", "480p", video_name)

    def get_first_seg_path(self, video_name, frame_list):
        return frame_list[0].replace("JPEGImages", "Annotations").replace("jpg", "png")

    def get_first_frame_output_name(self, frame_list):
        return "00000.png"

    def run_evaluation(self, results_path):
        script = os.path.join(self.module_dir, "davis2017-evaluation", "evaluation_method.py")
        cmd = [
            sys.executable, script,
            "--task", "semi-supervised",
            "--results_path", results_path,
            "--davis_path", self.data_path,
            "--set", self.subset,
        ]
        if self.video_list_file:
            cmd.extend(["--sequences_file", self.video_list_file])
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"DAVIS evaluation failed: {e}")
            raise


class YouTubeVOSHandler(DatasetHandler):
    """Handler for YouTube-VOS dataset."""

    def get_video_list(self):
        list_path = os.path.join(self.data_path, "ImageSets", f"{self.subset}.txt")
        return self._read_video_list(list_path)

    def get_video_dir(self, video_name):
        return os.path.join(self.data_path, "train", "JPEGImages", video_name)

    def get_first_seg_path(self, video_name, frame_list):
        ann_dir = os.path.join(self.data_path, "train", "Annotations", video_name)
        ann_files = sorted(glob.glob(os.path.join(ann_dir, "*.png")))
        return ann_files[0]

    def get_first_frame_output_name(self, frame_list):
        return frame_list[0].split("/")[-1].replace(".jpg", ".png")

    def run_evaluation(self, results_path):
        script = os.path.join(self.module_dir, "ytvos-evaluation", "evaluation_method.py")
        cmd = [
            sys.executable, script,
            "--task", "semi-supervised",
            "--results_path", results_path,
            "--ytvos_path", self.data_path,
            "--set", self.subset,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"YouTube-VOS evaluation failed: {e}")
            raise


class MOSEHandler(DatasetHandler):
    """Handler for MOSE dataset."""

    @staticmethod
    def _get_data_folder(subset):
        if subset in ("train", "val"):
            return "train"
        return "valid"

    def get_video_list(self):
        list_path = os.path.join(self.data_path, "ImageSets", f"{self.subset}.txt")
        return self._read_video_list(list_path)

    def get_video_dir(self, video_name):
        folder = self._get_data_folder(self.subset)
        return os.path.join(self.data_path, folder, "JPEGImages", video_name)

    def get_first_seg_path(self, video_name, frame_list):
        folder = self._get_data_folder(self.subset)
        return os.path.join(self.data_path, folder, "Annotations", video_name, "00000.png")

    def get_first_frame_output_name(self, frame_list):
        return "00000.png"

    def run_evaluation(self, results_path):
        script = os.path.join(self.module_dir, "mose-evaluation", "evaluation_method.py")
        cmd = [
            sys.executable, script,
            "--task", "semi-supervised",
            "--results_path", results_path,
            "--mose_path", self.data_path,
            "--set", self.subset,
        ]
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            logger.error(f"MOSE evaluation failed: {e}")
            raise


_HANDLER_MAP = {
    "davis": DAVISHandler,
    "ytvos": YouTubeVOSHandler,
    "mose": MOSEHandler,
}

SUPPORTED_DATASETS = set(_HANDLER_MAP.keys())


def get_handler(dataset, data_path, subset, module_dir, video_list_file=None):
    """Factory function to get the appropriate dataset handler."""
    cls = _HANDLER_MAP.get(dataset)
    if cls is None:
        raise ValueError(f"Unknown dataset '{dataset}'. Supported: {sorted(SUPPORTED_DATASETS)}")
    return cls(data_path, subset, module_dir, video_list_file=video_list_file)
