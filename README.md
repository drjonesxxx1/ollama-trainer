# ⚡ Ollama Personal Trainer & Unsloth MoE Orchestrator

[![Unsloth Accelerated](https://img.shields.io/badge/Unsloth-Triton%20Kernels-00F0FF?style=for-the-badge)](https://github.com/unslothai/unsloth)
[![Target Hardware](https://img.shields.io/badge/Hardware-RTX%204080%20Super%2016GB%20%2B%2064GB%20RAM-8A2BE2?style=for-the-badge)]()
[![Ollama Native](https://img.shields.io/badge/Ollama-Localhost%3A11434-00FF9D?style=for-the-badge)](https://ollama.ai)

An elite, full-stack **Studio Dashboard & Fine-Tuning Environment** for local Large Language Models (Qwen 2.5 Coder 32B, DeepSeek V3 671B MoE, Llama 3.1 8B).

Designed specifically for **Hardware-Aware MoE Expert Pruning**, **Harness-Grounded GRPO (Execution-in-the-Loop Reinforcement Learning)**, **MCP & ADB Plugin Synthesis**, and **Direct Ollama GGUF Deployment**.

---

## 📸 Core Features & Studio Modules

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       OLLAMA PERSONAL TRAINER WORKFLOW                       │
└─────────────────────────────────────────────────────────────────────────────┘
  1. Base Models    ──► Select Qwen-2.5-32B, DeepSeek-V3 671B MoE, Llama-3.1
  2. Techniques     ──► Configure LoRA (r=32), DoRA, FlashAttention-2, NF4
  3. Dataset Studio ──► Manage JSONL samples & Synthetic AI Generator
  4. MCP & ADB      ──► Declare Proxmox, nftables, dnsmasq & ADB tool schemas
  5. MoE Pruner     ──► Trace router gate activations & drop 70% trivia experts
  6. GRPO Training  ──► Real-time execution gym, multi-vector rewards, log stream
  7. GGUF Matrix    ──► Q4_K_M dual-offload quantization (16GB VRAM + 64GB RAM)
  8. Ollama Deploy  ──► One-click Modelfile build & Localhost:11434 push
  9. Arena Test     ──► Live interactive chat & real tool execution sandbox
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

### 3. Proxify-ADB Fleet Telemetry & MCP Plugins
- Pre-configured tool declarations for Proxmox Control Gateway (`10.30.20.1`), isolated bridge `vmbr1`, `dnsmasq` leases, `nftables` proxy routing rules, and ADB phone endpoints (`5555`).

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
├── server.ts                   # Express API, Vite middleware, Ollama proxy & Gitea sync
├── package.json                # React 19, Lucide, Recharts, Tailwind CSS v4, Express
├── vite.config.ts              # Vite bundle configuration
├── src/
│   ├── App.tsx                 # Master state controller & tab router
│   ├── components/
│   │   ├── Header.tsx          # Top HUD, VRAM load meter, Ollama connection badge
│   │   ├── ModelSelector.tsx   # Base model selector & hardware fit calculator
│   │   ├── TechniqueWorkshop.tsx # Hyperparameters, LoRA rank r, DoRA, NF4
│   │   ├── DatasetStudio.tsx   # JSONL dataset editor & Gemini synthetic generator
│   │   ├── MCPHarnessStudio.tsx# MCP tool schemas & ADB fleet commands
│   │   ├── PruningStudio.tsx   # MoE expert activation tracing & layer drop studio
│   │   ├── MoEStudio.tsx       # MoE merger & routing topology inspector
│   │   ├── TrainingSimulator.tsx # GRPO reward curves, live loss, cyber log stream
│   │   ├── GGUFStudio.tsx      # GGUF quantization matrix & system prompts
│   │   ├── OllamaDeployer.tsx  # Modelfile generator & Gitea push action
│   │   └── InteractiveArena.tsx# Live chat playground & tool call verification
│   ├── data/                   # Default models, MCP presets, hardware calculators
│   └── types.ts                # TypeScript interfaces
├── scripts/                    # Python pipeline execution scripts
│   ├── harvester.py            # Infrastructure Config Harvester
│   ├── prune_moe.py            # MoE Expert Profiler & Pruner
│   ├── harness_env.py          # Command Execution Safety Sandbox
│   ├── train_grpo.py           # Unsloth GRPO Trainer script
│   ├── deploy.py               # GGUF Export helper
│   └── state_eye.py            # 10.30.20.1 Telemetry Injector
└── infra_moe_grpo_blueprint.md # Complete technical architecture documentation
```

---

## 🔒 License

Licensed under the [MIT License](LICENSE). Developed for **drjones**.
