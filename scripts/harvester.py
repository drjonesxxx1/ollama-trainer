#!/usr/bin/env python3
"""
harvester.py - Model Variant Dataset & Instruction Curation Engine
Curates high-quality training pairs mapping natural language instructions to code and tool execution blocks.
"""

import json
import os
import sys
from typing import Dict, List, Any

OUTPUT_DATASET = "infra_dataset.jsonl"

def collect_infrastructure_configs() -> List[Dict[str, Any]]:
    """Generates synthetic dataset mapping instruction prompts to execution code blocks."""
    return [
        {
            "prompt": "Write a Python script to filter fine-tuning dataset samples by sequence length and quality score.",
            "completion": "```python\ndef filter_dataset(samples, min_len=20, min_score=0.8):\n    return [s for s in samples if len(s.get('output', '')) >= min_len and s.get('score', 1.0) >= min_score]\n```"
        },
        {
            "prompt": "Generate a SQL query to compute average token generation speed per model variant from evaluation metrics.",
            "completion": "```sql\nSELECT model_variant, AVG(tokens_per_sec) AS avg_speed, COUNT(*) AS eval_runs\nFROM benchmark_results\nGROUP BY model_variant;\n```"
        },
        {
            "prompt": "Write a Bash command to monitor GPU memory usage every 2 seconds and log output to a file.",
            "completion": "```bash\nnvidia-smi --query-gpu=timestamp,name,memory.used,memory.free,utilization.gpu --format=csv -l 2 > gpu_telemetry.log\n```"
        },
        {
            "prompt": "Format a Model Context Protocol (MCP) tool call payload for a code execution tool.",
            "completion": "```json\n{\n  \"tool\": \"python_interpreter\",\n  \"arguments\": {\n    \"code\": \"import torch\\nprint(torch.cuda.get_device_name(0))\"\n  }\n}\n```"
        }
    ]

def main():
    configs = collect_infrastructure_configs()
    with open(OUTPUT_DATASET, "w", encoding="utf-8") as f:
        for item in configs * 25:
            f.write(json.dumps(item) + "\n")
    print(f"[+] Generated {OUTPUT_DATASET} with {len(configs)*25} items.")

if __name__ == "__main__":
    main()

