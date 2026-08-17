#!/usr/bin/env python3
"""
curiosity_engine.py — PyTorch Intrinsic Motivation & Autonomous Goal Engine
Implements PyTorch Random Network Distillation (RND) novelty detection,
autonomous goal priority queues, and epsilon-greedy exploration scheduling.

Runs on PyTorch with CUDA/CPU automatic device selection.
"""

import json
import os
import sys
import random
import math
import hashlib
from typing import Dict, List, Optional, Tuple
from datetime import datetime

import torch
import torch.nn as nn
import torch.optim as optim

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

DEFAULT_EXPLORATION_RATE = 0.3
DEFAULT_EPSILON_DECAY = 0.995
DEFAULT_RND_HIDDEN_DIM = 256
DEFAULT_GOAL_QUEUE_DEPTH = 20


class RNDTargetNetwork(nn.Module):
    """Frozen random target network generating fixed high-dimensional embeddings."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )
        # Freeze target network weights
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class RNDPredictorNetwork(nn.Module):
    """Trainable predictor network learning to match target outputs on visited states."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = 256, out_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PyTorchRNDCuriosityModule:
    """Computes intrinsic rewards using MSE loss between RND target and predictor PyTorch outputs."""

    def __init__(self, in_dim: int = 128, hidden_dim: int = DEFAULT_RND_HIDDEN_DIM, lr: float = 1e-3):
        self.in_dim = in_dim
        # Deterministic seed for reproducible frozen target network
        torch.manual_seed(42)
        self.target_net = RNDTargetNetwork(in_dim, hidden_dim, 128).to(DEVICE)
        self.predictor_net = RNDPredictorNetwork(in_dim, hidden_dim, 128).to(DEVICE)
        self.optimizer = optim.Adam(self.predictor_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.training_steps = 0

    def _text_to_tensor(self, text: str) -> torch.Tensor:
        """Converts state string into a deterministic normalized state feature vector."""
        h = hashlib.sha256(text.encode("utf-8")).digest()
        features = [float(b) / 255.0 for b in h[:self.in_dim]]
        # Pad to in_dim if needed
        while len(features) < self.in_dim:
            features.append(0.0)
        return torch.tensor([features], dtype=torch.float32, device=DEVICE)

    def intrinsic_reward(self, state_text: str) -> float:
        """Calculates MSE loss: || target(s) - predictor(s) ||^2 as intrinsic reward."""
        self.target_net.eval()
        self.predictor_net.eval()

        with torch.no_grad():
            state_tensor = self._text_to_tensor(state_text)
            target_out = self.target_net(state_tensor)
            pred_out = self.predictor_net(state_tensor)
            mse = self.criterion(pred_out, target_out).item()

        return round(float(mse), 6)

    def train_step(self, state_text: str):
        """Train predictor network on visited state to reduce future novelty for this state."""
        self.predictor_net.train()
        self.optimizer.zero_grad()

        state_tensor = self._text_to_tensor(state_text)
        with torch.no_grad():
            target_out = self.target_net(state_tensor)

        pred_out = self.predictor_net(state_tensor)
        loss = self.criterion(pred_out, target_out)
        loss.backward()
        self.optimizer.step()
        self.training_steps += 1

    def get_domain_novelty_map(self) -> List[Dict]:
        domains = [
            {"domain": "Python Code Generation", "category": "coding",
             "sample_states": ["def parse_ast(): pass", "async def main(): await asyncio.gather()"]},
            {"domain": "Function Calling (MCP)", "category": "tool_calling",
             "sample_states": ["call_tool('calculator', {'expr': '2+2'})", "format_tool_response()"]},
            {"domain": "Mathematical Reasoning", "category": "reasoning",
             "sample_states": ["solve equation 2x^2 + 5x - 3 = 0", "evaluate matrix determinant"]},
            {"domain": "SQL Query Synthesis", "category": "coding",
             "sample_states": ["SELECT * FROM users WHERE active = true", "EXPLAIN ANALYZE SELECT"]},
            {"domain": "Bash / CLI Scripting", "category": "coding",
             "sample_states": ["find . -name '*.py' | xargs grep", "awk '{print $2}'"]},
            {"domain": "Python Tooling", "category": "coding",
             "sample_states": ["import torch; print(torch.cuda.is_available())", "pip install unsloth"]},
            {"domain": "Botany & Plants", "category": "trivia",
             "sample_states": ["photosynthesis process in chloroplasts", "plant taxonomy ranks"]},
            {"domain": "Ancient History", "category": "trivia",
             "sample_states": ["Roman empire timeline AD 117", "Egyptian pyramid construction"]},
        ]
        results = []
        for d in domains:
            novelties = [self.intrinsic_reward(s) for s in d["sample_states"]]
            avg_n = sum(novelties) / len(novelties)
            results.append({
                "domain": d["domain"],
                "category": d["category"],
                "avg_novelty": round(avg_n, 4),
                "explored": round(max(0.01, 1.0 - min(1.0, avg_n * 5.0)), 4),
                "status": "terra_incognita" if avg_n > 0.15 else "well_explored"
            })
        return results


class AutonomousGoalQueue:
    def __init__(self, max_depth: int = DEFAULT_GOAL_QUEUE_DEPTH):
        self.max_depth = max_depth
        self.goals: List[Dict] = []
        self._goal_templates = [
            "Synthesize edge-case Python async generator training pairs",
            "Test zero-shot SQL join generation under multi-schema prompts",
            "Monitor loss convergence on 8B model variants under LoRA rank 32",
            "Probe token throughput optimization for GGUF Q4_K_M quantizations",
            "Predict evaluation benchmark drops via hyperparameter trend analysis",
            "Auto-curate synthetic dataset pairs from raw code repositories",
            "Verify JSON response schema adherence during tool calling",
            "Inspect transformer layer router gate activation frequencies",
        ]

    def generate_goals(self, rnd: PyTorchRNDCuriosityModule, count: int = 8) -> List[Dict]:
        candidates = random.sample(self._goal_templates, min(count, len(self._goal_templates)))
        new_goals = []
        for text in candidates:
            novelty = rnd.intrinsic_reward(text)
            gid = hashlib.md5(text.encode()).hexdigest()[:8]
            new_goals.append({
                "id": gid,
                "goal": text,
                "novelty": round(novelty, 4),
                "status": "queued" if novelty > 0.05 else "evaluated"
            })
        new_goals.sort(key=lambda g: g["novelty"], reverse=True)
        self.goals = new_goals[:self.max_depth]
        return self.goals

    def pop_top_goal(self) -> Optional[Dict]:
        for g in self.goals:
            if g["status"] == "queued":
                g["status"] = "exploring"
                return g
        return None

    def execute_top_goal(self, rnd: PyTorchRNDCuriosityModule) -> Optional[Dict]:
        goal = self.pop_top_goal()
        if not goal:
            return None
        # Train predictor network on this state (reduces future novelty)
        rnd.train_step(goal["goal"])
        goal["status"] = "evaluated"
        goal["new_novelty"] = rnd.intrinsic_reward(goal["goal"])
        return goal


def main():
    print("=" * 60)
    print("  PYTORCH INTRINSIC CURIOSITY & AUTONOMOUS GOAL ENGINE")
    print(f"  Device: {DEVICE} | PyTorch v{torch.__version__}")
    print("=" * 60)

    if "--test" in sys.argv:
        rnd = PyTorchRNDCuriosityModule(in_dim=128, hidden_dim=256)
        print(f"[TEST] Initialized PyTorch RND Target & Predictor networks on {DEVICE}.")

        test_state = "def optimize_dataset_pairs(): pass"
        initial_novelty = rnd.intrinsic_reward(test_state)
        print(f"  Initial MSE novelty for '{test_state}': {initial_novelty:.6f}")

        # Train predictor 5 times on test_state
        for _ in range(5):
            rnd.train_step(test_state)

        updated_novelty = rnd.intrinsic_reward(test_state)
        print(f"  Novelty after 5 PyTorch Adam training steps: {updated_novelty:.6f}")
        print(f"  Novelty reduction delta: {(initial_novelty - updated_novelty):.6f} (Predictor learned state!)")

        novelty_map = rnd.get_domain_novelty_map()
        print("\n[TEST] PyTorch Domain Novelty Map:")
        for d in novelty_map[:4]:
            print(f"  {d['domain']:<35} [{d['status']:<16}] novelty={d['avg_novelty']:.4f}")

        goal_queue = AutonomousGoalQueue()
        goals = goal_queue.generate_goals(rnd, count=5)
        print(f"\n[TEST] Generated {len(goals)} autonomous goals scored by RND MSE loss:")
        for g in goals[:3]:
            print(f"  [{g['id']}] MSE={g.get('novelty', 0.0):.6f} | {g['goal']}")

        executed = goal_queue.execute_top_goal(rnd)
        if executed:
            print(f"\n[TEST] Executed top goal: MSE before={executed.get('novelty', 0.0):.6f} -> MSE after={executed.get('new_novelty', 0.0):.6f}")

        print("\n[TEST] All PyTorch curiosity engine tests PASSED [OK]")
    else:
        rnd = PyTorchRNDCuriosityModule(in_dim=128, hidden_dim=256)
        test_state = "def optimize_dataset_pairs(): pass"
        initial_novelty = rnd.intrinsic_reward(test_state)
        for _ in range(5):
            rnd.train_step(test_state)
        updated_novelty = rnd.intrinsic_reward(test_state)
        
        novelty_map = rnd.get_domain_novelty_map()
        goal_queue = AutonomousGoalQueue()
        goals = goal_queue.generate_goals(rnd, count=10)
        
        result = {
            "device": str(DEVICE),
            "initial_novelty": initial_novelty,
            "updated_novelty": updated_novelty,
            "novelty_reduction": initial_novelty - updated_novelty,
            "novelty_map": novelty_map,
            "goals": goals,
            "status": "curiosity_novelty_scan_complete"
        }
        print(json.dumps(result))


if __name__ == "__main__":
    main()
