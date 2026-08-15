# ⚡ Ollama Personal Trainer & Unsloth MoE Orchestrator

[![Unsloth Accelerated](https://img.shields.io/badge/Unsloth-Triton%20Kernels-00F0FF?style=for-the-badge)](https://github.com/unslothai/unsloth)
[![Target Hardware](https://img.shields.io/badge/Hardware-RTX%204080%20Super%2016GB%20%2B%2064GB%20RAM-8A2BE2?style=for-the-badge)]()
[![Ollama Native](https://img.shields.io/badge/Ollama-Localhost%3A11434-00FF9D?style=for-the-badge)](https://ollama.ai)

An elite, full-stack **Studio Dashboard & Fine-Tuning Environment** for local Large Language Models (Qwen 2.5 Coder 32B, DeepSeek V3 671B MoE, Llama 3.1 8B).

Designed specifically for **Hardware-Aware MoE Expert Pruning**, **Harness-Grounded GRPO (Execution-in-the-Loop Reinforcement Learning)**, **MCP Tool & Function Calling Synthesis**, and **Direct Ollama GGUF Deployment**.

---

## 📸 Core Features & Studio Modules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                      OLLAMA PERSONAL TRAINER WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘
  1. Base Models    ──► Select Qwen-2.5-32B, DeepSeek-V3 671B MoE, Llama-3.1
  2. Human Knobs    ──► Shed weight (70% trivia experts) & set tool obsession
  3. GPU Pipeline   ──► Unsloth GRPO RL on NVIDIA RTX 4080 Super GPU
  4. Test Drive     ──► Interactive model variant chat & code execution sandbox
  5. Plasticity     ──► Elastic Weight Consolidation (EWC) + Episodic Replay
  6. Curiosity      ──► Random Network Distillation (RND) & Autonomous Goal Queue
  7. World Model    ──► Latent state prediction & Monte Carlo Tree Search (MCTS)
  8. Metacognition  ──► Confidence calibration (ECE) & NAS Lite (Thompson Sampling)
```

---

## 🛠️ Deep Subsystem Integration

### 1. MoE Router Activation Profiling & Expert Weight Pruning
- **Domain-Specific Tracing**: Traces top-k router gate selection frequencies across all 61 transformer layers during calibration forward passes.
- **Surgical Expert Dropping**: Drops dormant experts holding botany, literature, or trivia knowledge (256 experts $\rightarrow$ 64 experts), saving ~68.75% parameter weight (~34.5 GB System RAM + 6.2 GB VRAM in Q4_K_M).

### 2. Harness-Grounded GRPO (Execution-in-the-Loop RL)
- **Hard Execution Rewards ($R_{\text{exec}}$)**: $+3.0$ for exit code 0, $-1.5$ for runtime exceptions, $-10.0$ for safety violations.
- **Anti-Hesitation Penalty ($R_{\text{anti\_hesit}}$)**: $+2.0$ if the command code block is initiated within 25 tokens, suppressing natural language chatter.
- **Rolling Cyber Terminal**: Live WebSocket log console with level filtering (`INFO`, `HARNESS`, `REWARD`, `WARN`, `ERROR`) and auto-scroll controls.

### 3. AGI Paradigm Integration (Steps 5–8)
- **Continuous Plasticity (Step 5)**: EWC diagonal Fisher penalty keeps old-task retention high while adapting to live interactions via episodic replay (`episodic_memory.jsonl`) and auto LoRA hot-swap into Ollama.
- **Intrinsic Curiosity (Step 6)**: RND target/predictor error measures novelty; autonomous goal priority queue explores uncharted infrastructure states.
- **Causal World Model (Step 7)**: Latent state prediction + MCTS rollouts plan multi-step action sequences; counterfactual simulator predicts risk before command execution.
- **Metacognitive Self-Refinement (Step 8)**: Expected Calibration Error (ECE) minimization, knowledge consistency audit, Thompson Sampling NAS lite, and automatic self-correction loops.

---

## 🚀 Quickstart Guide

### 1. Requirements
- **OS**: Windows 11 / Linux (WSL2 Ubuntu 24.04 recommended for Triton acceleration)
- **Node.js**: v18.x or higher (v24 tested)
- **Ollama**: Running locally on `http://localhost:11434`
- **GPU**: NVIDIA RTX 4080 Super (16 GB VRAM) + 64 GB System RAM

### 2. Installation
```bash
git clone https://gitea.thetempleofdoom.com/drjones/ollama-personal-trainer.git
cd ollama-personal-trainer
npm install
```

### 3. Launching Studio
```bash
# Start full-stack React + Express server (Runs on http://localhost:3000)
npm run dev
```

### 4. Production Build & Server Start
```bash
npm run build
npm start
```

---

## 📦 Project File Structure

```
ollama-personal-trainer/
├── server.ts                   # Express API, Vite middleware, Ollama proxy & AGI endpoints
├── package.json                # React 19, Lucide, Recharts, Tailwind CSS v4, Express
├── vite.config.ts              # Vite bundle configuration
├── src/
│   ├── App.tsx                 # Master state controller & 8-step tab router
│   ├── components/
│   │   ├── Header.tsx          # Top HUD, VRAM load meter, 8-step nav bar
│   │   ├── EasyPipelineWizard.tsx # Master 8-step assembly line wizard
│   │   ├── OnlineLearningPanel.tsx # Step 5: Continuous Plasticity (EWC & Replay)
│   │   ├── CuriosityEnginePanel.tsx # Step 6: Intrinsic Curiosity (RND & Goal Queue)
│   │   ├── WorldModelPanel.tsx    # Step 7: Causal World Model & MCTS Planner
│   │   ├── MetacognitionPanel.tsx # Step 8: Metacognitive Self-Refinement Engine
│   │   └── OllamaTerminalModal.tsx# CLI Terminal & Ollama model manager
│   ├── data/                   # Default models, MCP presets, hardware calculators
│   └── types.ts                # TypeScript interfaces (including AGI configs)
├── scripts/                    # Python pipeline execution scripts
├── online_learner.py           # EWC + Episodic Replay + LoRA Hot-Swap
├── curiosity_engine.py         # RND Novelty + Autonomous Goal Priority Queue
├── world_model.py              # Latent State Predictor + MCTS + Counterfactual
├── metacognition.py            # Confidence Calibration + Consistency Audit + NAS
├── harvester.py                # Infrastructure Config Harvester
├── prune_moe.py                # MoE Expert Profiler & Pruner
├── harness_env.py              # Command Execution Safety Sandbox
├── train_grpo.py               # Unsloth GRPO Trainer script
└── deploy.py                   # GGUF Export helper
```

---

## 🔒 License

Licensed under the [MIT License](LICENSE). Developed for **drjones**.
