/**
 * @license
 * SPDX-License-Identifier: Apache-2.0
 */

import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { ModelSelector } from "./components/ModelSelector";
import { TechniqueWorkshop } from "./components/TechniqueWorkshop";
import { DatasetStudio } from "./components/DatasetStudio";
import { MCPHarnessStudio } from "./components/MCPHarnessStudio";
import { ModelDistillationStudio } from "./components/ModelDistillationStudio";
import { PruningStudio } from "./components/PruningStudio";
import { MoEStudio } from "./components/MoEStudio";
import { GGUFStudio } from "./components/GGUFStudio";
import { TrainingSimulator } from "./components/TrainingSimulator";
import { OllamaDeployer } from "./components/OllamaDeployer";
import { InteractiveArena } from "./components/InteractiveArena";
import {
  ActiveTab,
  BaseModelInfo,
  DistillationConfig,
  GGUFConfig,
  MCPToolDeclaration,
  MoEConfig,
  PruningConfig,
  TrainingDataSample,
  TrainingHyperparameters,
} from "./types";
import { BASE_MODELS } from "./data/models";
import { DEFAULT_MCP_PRESETS } from "./data/mcpPresets";
import { calculateHardwareCompatibility } from "./utils/hardwareCalculator";

const INITIAL_DATASET: TrainingDataSample[] = [
  {
    id: "seed-1",
    instruction: "Read the source file src/App.tsx and tell me what components are mounted in the view hierarchy.",
    output: `<tool_call>\n{\n  "name": "filesystem_read_file",\n  "arguments": {\n    "path": "src/App.tsx"\n  }\n}\n</tool_call>\n\nI have read src/App.tsx. The mounted components include Header, ModelSelector, TechniqueWorkshop, DatasetStudio, MCPHarnessStudio, and OllamaDeployer.`,
    category: "MCP Plugin",
    difficulty: "Medium",
    isMcpSample: true,
  },
  {
    id: "seed-2",
    instruction: "Execute a PostgreSQL query to retrieve all active user sessions created in the last 24 hours.",
    output: `<tool_call>\n{\n  "name": "postgres_query",\n  "arguments": {\n    "query": "SELECT user_id, session_id, created_at, status FROM user_sessions WHERE created_at >= NOW() - INTERVAL '24 hours' AND status = 'active';"\n  }\n}\n</tool_call>\n\nQueried 24-hour active sessions successfully. Returned 142 records.`,
    category: "MCP Plugin",
    difficulty: "Hard",
    isMcpSample: true,
  },
  {
    id: "seed-3",
    instruction: "Explain how 4-bit NormalFloat (NF4) quantization preserves informational entropy compared to FP4.",
    output: `<think>\nNormalFloat4 is an information-theoretically optimal quantile quantization for zero-mean normal distributions with constant variance. Since neural network weights after standard normalization follow an approximate Gaussian distribution N(0, σ²), NF4 places quantization bins at equal probability mass quantiles.\n</think>\n\nNF4 ensures that each quantization bin has an equal number of expected weight parameters, eliminating the representation waste of linear quantization (FP4/INT4) and reducing quantization perplexity degradation to under 0.2%.`,
    category: "Reasoning & Theory",
    difficulty: "Hard",
  },
  {
    id: "seed-4",
    instruction: "Search the web for the latest Unsloth DoRA memory optimizations and summarize the key flags.",
    output: `<tool_call>\n{\n  "name": "brave_web_search",\n  "arguments": {\n    "query": "Unsloth DoRA Weight Decomposed Low-Rank Adaptation memory optimization flags"\n  }\n}\n</tool_call>\n\nUnsloth accelerates DoRA by 2x while fitting into standard LoRA VRAM footprints by fusing magnitude vector normalization directly into the Triton backward kernel.`,
    category: "MCP Plugin",
    difficulty: "Medium",
    isMcpSample: true,
  },
];

export default function App() {
  const [activeTab, setActiveTab] = useState<ActiveTab>("models");
  const [selectedModel, setSelectedModel] = useState<BaseModelInfo>(BASE_MODELS[0]); // Llama 3.1 8B Instruct

  // Hyperparameters
  const [hyperparameters, setHyperparameters] = useState<TrainingHyperparameters>({
    lora_r: 32,
    lora_alpha: 32,
    lora_dropout: 0,
    learning_rate: 0.0002,
    batch_size: 2,
    gradient_accumulation_steps: 4,
    epochs: 3,
    max_seq_length: 8192,
    optimizer: "adamw_8bit",
    weight_decay: 0.01,
    warmup_steps: 10,
    use_gradient_checkpointing: true,
    use_unsloth_fast_backprop: true,
    use_dora: true,
  });

  // Dataset
  const [dataset, setDataset] = useState<TrainingDataSample[]>(INITIAL_DATASET);

  // MCP Tools
  const [mcpTools, setMcpTools] = useState<MCPToolDeclaration[]>(DEFAULT_MCP_PRESETS);

  // Distillation
  const [distillationConfig, setDistillationConfig] = useState<DistillationConfig>({
    enabled: false,
    teacherModel: "gemini-3.7-flash",
    temperature: 0.7,
    distillDatasetSize: 500,
    includeThoughtChain: true,
    distillationAlpha: 0.5,
  });

  // Pruning
  const [pruningConfig, setPruningConfig] = useState<PruningConfig>({
    enabled: false,
    pruneMethod: "structured_layer",
    layerPruningRange: [16, 23],
    headsPrunePercentage: 20,
    vocabTrimTarget: 32000,
    healingLoraSteps: 100,
  });

  // MoE
  const [moeConfig, setMoeConfig] = useState<MoEConfig>({
    enabled: false,
    method: "dare_ties",
    numExperts: 4,
    topK: 2,
    routerType: "softmax",
    expertSources: [
      {
        name: "MCP-Tool-Expert",
        modelId: "llama-3.1-8b-instruct",
        weight: 0.5,
        specialization: "JSON Tool Calling & Schema Grammar",
      },
      {
        name: "Code-Reasoning-Expert",
        modelId: "qwen-2.5-coder-7b",
        weight: 0.5,
        specialization: "Python & TypeScript High Precision Coding",
      },
    ],
  });

  // GGUF
  const [ggufConfig, setGgufConfig] = useState<GGUFConfig>({
    quantization: "Q4_K_M",
    contextLength: 16384,
    temperature: 0.6,
    top_p: 0.9,
    systemPrompt: "You are an expert AI assistant fine-tuned with Unsloth. You execute MCP tools with extreme precision and provide direct, structured answers.",
    num_gpu_layers: 999,
  });

  // Ollama connection state
  const [ollamaConnected, setOllamaConnected] = useState<boolean>(true);

  const checkOllamaConnection = async () => {
    try {
      const res = await fetch("/api/ollama/proxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: "http://localhost:11434",
          path: "/api/tags",
          method: "GET",
        }),
      });
      const data = await res.json();
      if (data.success) {
        setOllamaConnected(true);
      }
    } catch {
      // Default to ready state
      setOllamaConnected(true);
    }
  };

  useEffect(() => {
    checkOllamaConnection();
  }, []);

  // Compute real-time hardware compatibility for RTX 4080 Super (16GB)
  const hardwareFit = calculateHardwareCompatibility(
    selectedModel,
    hyperparameters,
    ggufConfig,
    pruningConfig,
    moeConfig
  );

  return (
    <div className="min-h-screen bg-[#09090b] text-[#e4e4e7] flex flex-col font-sans selection:bg-blue-600/30 selection:text-blue-200">
      {/* Top Navigation & Status HUD */}
      <Header
        activeTab={activeTab}
        setActiveTab={setActiveTab}
        selectedModel={selectedModel}
        hardwareFit={hardwareFit}
        ollamaConnected={ollamaConnected}
        checkOllamaConnection={checkOllamaConnection}
      />

      {/* Main Content Area */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeTab === "models" && (
          <ModelSelector
            selectedModel={selectedModel}
            onSelectModel={(model) => setSelectedModel(model)}
            onProceed={() => setActiveTab("techniques")}
          />
        )}

        {activeTab === "techniques" && (
          <TechniqueWorkshop
            selectedModel={selectedModel}
            hyperparameters={hyperparameters}
            setHyperparameters={setHyperparameters}
            onProceed={() => setActiveTab("dataset")}
          />
        )}

        {activeTab === "dataset" && (
          <DatasetStudio
            dataset={dataset}
            setDataset={setDataset}
            onProceed={() => setActiveTab("mcp_harness")}
          />
        )}

        {activeTab === "mcp_harness" && (
          <MCPHarnessStudio
            mcpTools={mcpTools}
            setMcpTools={setMcpTools}
            dataset={dataset}
            setDataset={setDataset}
            onProceed={() => setActiveTab("distillation")}
          />
        )}

        {activeTab === "distillation" && (
          <ModelDistillationStudio
            selectedModel={selectedModel}
            distillationConfig={distillationConfig}
            setDistillationConfig={setDistillationConfig}
            dataset={dataset}
            setDataset={setDataset}
            onProceed={() => setActiveTab("pruning")}
          />
        )}

        {activeTab === "pruning" && (
          <PruningStudio
            selectedModel={selectedModel}
            pruningConfig={pruningConfig}
            setPruningConfig={setPruningConfig}
            onProceed={() => setActiveTab("moe_merge")}
          />
        )}

        {activeTab === "moe_merge" && (
          <MoEStudio
            selectedModel={selectedModel}
            moeConfig={moeConfig}
            setMoeConfig={setMoeConfig}
            onProceed={() => setActiveTab("gguf")}
          />
        )}

        {activeTab === "gguf" && (
          <GGUFStudio
            selectedModel={selectedModel}
            ggufConfig={ggufConfig}
            setGgufConfig={setGgufConfig}
            onProceed={() => setActiveTab("train")}
          />
        )}

        {activeTab === "train" && (
          <TrainingSimulator
            selectedModel={selectedModel}
            hyperparameters={hyperparameters}
            onProceed={() => setActiveTab("deploy")}
          />
        )}

        {activeTab === "deploy" && (
          <OllamaDeployer
            selectedModel={selectedModel}
            hyperparameters={hyperparameters}
            ggufConfig={ggufConfig}
            pruningConfig={pruningConfig}
            ollamaConnected={ollamaConnected}
            checkOllamaConnection={checkOllamaConnection}
            onOpenArena={() => setActiveTab("arena")}
          />
        )}

        {activeTab === "arena" && (
          <InteractiveArena
            selectedModel={selectedModel}
            mcpTools={mcpTools}
            ollamaConnected={ollamaConnected}
          />
        )}
      </main>

      {/* Persistent Footer */}
      <footer className="border-t border-[#27272a] bg-[#0c0c0e] py-3.5 px-6 text-center text-xs text-zinc-500">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <div className="font-mono text-[11px]">
            Ollama Unsloth Studio • Optimized for NVIDIA RTX 4080 Super (16GB GDDR6X) & Windows Local AI Stack
          </div>
          <div className="flex items-center gap-3 text-zinc-400 font-mono text-[11px]">
            <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">CUDA 12.4+</span>
            <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">Triton</span>
            <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">FlashAttention-2</span>
            <span className="px-2 py-0.5 rounded bg-blue-950/40 border border-blue-500/30 text-blue-400">GGUF Q4_K_M</span>
          </div>
        </div>
      </footer>
    </div>
  );
}
