"""vLLM-endpoint backend for Level 1 transition detection.

Same OpenAI-compatible client + guided-JSON pattern used across all three
sibling pipelines now. Level 1's response is an array of transition frames
(a window may contain several, or none) — simpler than Level 2's schema
since there is no action/hand/object/subtask here, only frame + short
before/after state description.
"""

import urllib.request

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "transitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "frame": {"type": "integer"},
                    "before": {"type": "string", "maxLength": 60},
                    "after": {"type": "string", "maxLength": 60},
                },
                "required": ["frame", "before", "after"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["transitions"],
    "additionalProperties": False,
}


class EndpointBackend:
    def __init__(self, cfg):
        from openai import OpenAI
        v, lv1 = cfg["vlm"], cfg["level1_transitions"]
        self.client = OpenAI(base_url=v["endpoint_url"], api_key=v["api_key"])
        self.model = v["model_name"]
        self.temperature = lv1["temperature"]
        self.max_tokens = lv1["max_new_tokens"]

    def generate(self, content):
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": content}],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            response_format={"type": "json_schema", "json_schema": {
                "name": "window_transitions", "schema": RESPONSE_SCHEMA}},
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
            f"'{{\"image\": {cfg['level1_transitions']['keyframes_per_window']}}}' "
            f"--gpu-memory-utilization 0.92 --port 8000")
