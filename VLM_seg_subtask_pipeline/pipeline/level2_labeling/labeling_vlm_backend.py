"""vLLM-endpoint backend for Level 2 segment labeling.

Same OpenAI-compatible client + guided-JSON pattern used across all three
sibling pipelines. Unlike Kinematics_pipeline's Level 2 schema, `hand` is
part of THIS schema (there is no pose signal to compute it from here, so
the VLM is asked for it directly) and there is no `hand_disagreement` field
(nothing to disagree with).
"""

import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from common import ACTIONS, CONFIDENCES, HANDS  # noqa: E402

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": ACTIONS},
        "hand": {"type": "string", "enum": HANDS},
        "object": {"type": "string", "maxLength": 60},
        "subtask": {"type": "string", "maxLength": 200},
        "confidence": {"type": "string", "enum": CONFIDENCES},
    },
    "required": ["action", "hand", "object", "subtask", "confidence"],
    "additionalProperties": False,
}


class EndpointBackend:
    def __init__(self, cfg):
        from openai import OpenAI
        v, lv2 = cfg["vlm"], cfg["level2_labeling"]
        self.client = OpenAI(base_url=v["endpoint_url"], api_key=v["api_key"])
        self.model = v["model_name"]
        self.temperature = lv2["temperature"]
        self.max_tokens = lv2["max_new_tokens"]

    def generate(self, content):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "segment_label", "schema": RESPONSE_SCHEMA}},
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
            f"'{{\"image\": {cfg['level2_labeling']['keyframes_per_segment']}}}' "
            f"--gpu-memory-utilization 0.92 --port 8000")
