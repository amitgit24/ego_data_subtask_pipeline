"""vLLM-endpoint backend for the non-overlapping chunked generator
(generate_subtasks_chunked.py) -- a parallel sibling to vlm_backend.py, NOT
a modification of it. A chunk's output shape is identical to a sliding
window's (an array of labeled subtasks), so RESPONSE_SCHEMA and
check_vlm_endpoint are imported and reused as-is; only the client reads
temperature/max_tokens from THIS pipeline's own config_chunked.yaml
(cfg["generate"] there, a different file from the sliding-window sibling's
cfg["generate"]), since the two generators are tuned/swept independently.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from vlm_backend import RESPONSE_SCHEMA, check_vlm_endpoint  # noqa: E402,F401


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
                "name": "chunk_subtasks", "schema": RESPONSE_SCHEMA}},
        )
        return resp.choices[0].message.content
