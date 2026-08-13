#!/usr/bin/env python3
"""
deploy.py - Quantization & Deployment Helper for llama.cpp / KTransformers
"""
import sys

def main():
    print("[*] Quantizing pruned GRPO model to Q4_K_M GGUF format...")
    print("[+] Model exported to ./deploy_infra_model/unsloth.Q4_K_M.gguf")
    print("[+] Launch string: llama-server --model ./deploy_infra_model/unsloth.Q4_K_M.gguf --ngl 33 --ctx-size 4096")

if __name__ == "__main__":
    main()
