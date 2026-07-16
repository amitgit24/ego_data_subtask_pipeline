"""vLLM-endpoint backend for single-pass window generation.

Same OpenAI-compatible client + guided-JSON pattern proven in
Kinematics_pipeline/pipeline/level2_vlm/label_segments.py, adapted for a
response that is an ARRAY of subtasks (a window can contain several) rather
than one label per request. Structured decoding (vLLM/xgrammar) enforces the
schema at generation time, so a malformed field is impossible, not just
retried.
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ACTIONS, CONFIDENCES, HANDS  # noqa: E402

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "subtasks": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "start_frame": {"type": "integer"},
                    "end_frame": {"type": "integer"},
                    "action": {"type": "string", "enum": ACTIONS},
                    "hand": {"type": "string", "enum": HANDS},
                    "object": {"type": "string", "maxLength": 60},
                    "subtask": {"type": "string", "maxLength": 200},
                    "confidence": {"type": "string", "enum": CONFIDENCES},
                    "ongoing": {"type": "boolean"},
                },
                "required": ["start_frame", "end_frame", "action", "hand",
                            "object", "subtask", "confidence", "ongoing"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["subtasks"],
    "additionalProperties": False,
}


class EndpointBackend:
    def __init__(self, cfg):
        from openai import OpenAI
        v, g = cfg["vlm"], cfg["generate"]
        self.client = OpenAI(base_url=v["endpoint_url"], api_key=v["api_key"])
        self.model = v["model_name"]
        self.temperature = g["temperature"]
        self.max_tokens = g["max_new_tokens"]

    def generate(self, content):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "window_subtasks", "schema": RESPONSE_SCHEMA}},
        )
        return resp.choices[0].message.content


def check_vlm_endpoint(cfg):
    url = cfg["vlm"]["endpoint_url"].rstrip("/") + "/models"
    try:
        urllib.request.urlopen(url, timeout=5)
    except Exception as e:
        raise SystemExit(
            f"vLLM endpoint {url} unreachable ({e}). Start it with:\n"
            f"  CUDA_VISIBLE_DEVICES=0 VLLM_USE_FLASHINFER_SAMPLER=0 "
            f"vllm serve {cfg['paths']['model_path']} "
            f"--served-model-name {cfg['vlm']['model_name']} "
            f"--max-model-len 24576 "
            f"--limit-mm-per-prompt "
            f"'{{\"image\": {cfg['vlm']['keyframes_per_window']}}}' "
            f"--gpu-memory-utilization 0.92 --port 8000")
