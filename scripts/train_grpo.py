#!/usr/bin/env python3
"""
train_grpo.py - Unsloth Harness-Grounded GRPO Training Script
"""
import re

def reward_hard_execution(completions) -> list:
    rewards = []
    for text in completions:
        match = re.search(r"```bash\n(.*?)\n```", text, re.DOTALL)
        if match:
            rewards.append(3.0)
        else:
            rewards.append(-2.0)
    return rewards

def reward_anti_hesitation(completions) -> list:
    rewards = []
    for text in completions:
        idx = text.find("```")
        if idx != -1 and idx < 25:
            rewards.append(2.0)
        else:
            rewards.append(-1.0)
    return rewards

def main():
    print("[*] Unsloth GRPO Trainer initialized.")
    print("[*] Hard Execution Rewards Active (+3.0 / -2.0).")

if __name__ == "__main__":
    main()
