import os
import numpy as np
from PIL import Image
from realm.inference.base import BaseInferenceClient

class VJEPAClient(BaseInferenceClient):
    """
    Client for VJEPA ACTION Model inference
    """
    def __init__(self, host="127.0.0.1", port=8000, timeout_ms=15000):
        # 初始化基于 ZeroMQ 的基础客户端对象
        super().__init__(host=host, port=port, timeout_ms=timeout_ms)

    def infer(self, obs_dict):
        """
        调用远端 Server 的 get_action endpoint。
        """
        return self.call_endpoint("get_action", data=obs_dict)

    def set_goal(self, goal_image, goal_state=None):
        """
        设置目标状态，包括目标图像（必须）和可选的物理状态张量
        """
        payload = {
            "goal_image": goal_image,
        }
        if goal_state is not None:
            payload["goal_state"] = goal_state
        response = self.call_endpoint("set_goal", data=payload)
        return response

    def plot_pca(self, save_path, timeout_ms=600000):
        payload = {"save_path": save_path}
        old_timeout_ms = self.timeout_ms
        if timeout_ms is not None:
            self.set_timeout(timeout_ms)
        try:
            response = self.call_endpoint("plot_pca", data=payload)
        finally:
            if timeout_ms is not None:
                self.set_timeout(old_timeout_ms)
        img_bytes = response.get("image_bytes", b"")
        if img_bytes:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, "wb") as f:
                f.write(img_bytes)
        return response

    def plot_slot_masks(self, save_path):
        payload = {"save_path": save_path}
        response = self.call_endpoint("plot_slot_masks", data=payload)
        if response.get("error"):
            raise RuntimeError(response["error"])
        img_bytes = response.get("image_bytes", b"")
        if not img_bytes:
            raise RuntimeError(f"plot_slot_masks returned no image bytes: {response}")
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, "wb") as f:
            f.write(img_bytes)
        return response

    def render_current_slot_mask_panels(self, payload):
        response = self.call_endpoint("render_current_slot_mask_panels", data=payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def render_goal_slot_mask_panels(self):
        response = self.call_endpoint("render_goal_slot_mask_panels", data={})
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def get_debug_trace(self):
        return self.call_endpoint("get_debug_trace", requires_input=False)

    def get_warmup_config(self):
        return self.call_endpoint("get_warmup_config", requires_input=False)

    def score_observations_to_goal(self, payload):
        response = self.call_endpoint("score_observations_to_goal", data=payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def advance_goal(self, goal_idx=None):
        payload = {}
        if goal_idx is not None:
            payload["goal_idx"] = int(goal_idx)
        response = self.call_endpoint("advance_goal", data=payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def record_step_with_action(self, payload):
        response = self.call_endpoint("record_step_with_action", data=payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def predict_one_step_candidates(self, payload):
        response = self.call_endpoint("predict_one_step_candidates", data=payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response

    def record_one_step_prediction_post(self, payload):
        response = self.call_endpoint("record_one_step_prediction_post", data=payload)
        if isinstance(response, dict) and response.get("error"):
            raise RuntimeError(response["error"])
        return response
