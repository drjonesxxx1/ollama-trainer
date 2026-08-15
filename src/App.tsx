import React, { useState, useEffect } from "react";
import { Header } from "./components/Header";
import { EasyPipelineWizard } from "./components/EasyPipelineWizard";
import { OllamaTerminalModal } from "./components/OllamaTerminalModal";
import { DishwasherModal } from "./components/DishwasherModal";
import { DatasetStudio } from "./components/DatasetStudio";
import { MCPHarnessStudio } from "./components/MCPHarnessStudio";
import { ModelDistillationStudio } from "./components/ModelDistillationStudio";
import { TechniqueWorkshop } from "./components/TechniqueWorkshop";
import { TrainingSimulator } from "./components/TrainingSimulator";
import { MoEStudio } from "./components/MoEStudio";
import { PruningStudio } from "./components/PruningStudio";
import { GGUFStudio } from "./components/GGUFStudio";
import { OllamaDeployer } from "./components/OllamaDeployer";
import { InteractiveArena } from "./components/InteractiveArena";
import { BaseModelInfo, TrainingDataSample, MCPToolDeclaration } from "./types";
import { BASE_MODELS } from "./data/models";
import { DEFAULT_MCP_PRESETS } from "./data/mcpPresets";
import { calculateHardwareCompatibility } from "./utils/hardwareCalculator";

export default function App() {
  const [activeMode, setActiveMode] = useState<"pipeline" | "studio">("pipeline");
  const [activeStudioTab, setActiveStudioTab] = useState<string>("data_mcp");
  const [dataSubTab, setDataSubTab] = useState<"synthetic" | "mcp" | "distillation">("synthetic");
  const [pruneSubTab, setPruneSubTab] = useState<"moe" | "layer">("moe");
  const [deploySubTab, setDeploySubTab] = useState<"gguf" | "ollama">("gguf");

  const [currentStep, setCurrentStep] = useState<number>(1);
  const [selectedModel, setSelectedModel] = useState<BaseModelInfo>(BASE_MODELS[0]); // Qwen 2.5 Coder 32B / Llama 3.1
  const [ollamaConnected, setOllamaConnected] = useState<boolean>(true);
  const [isOllamaManagerOpen, setIsOllamaManagerOpen] = useState<boolean>(false);
  const [isDishwasherOpen, setIsDishwasherOpen] = useState<boolean>(false);

  // Global shared state
  const [dataset, setDataset] = useState<TrainingDataSample[]>([
    {
      id: "sample-1",
      instruction: "Write a Python function to parse JSON telemetry lines and calculate average CPU utilization.",
      input: "logs = ['{\"cpu\": 45.2}', '{\"cpu\": 52.1}']",
      output: "```python\nimport json\n\ndef avg_cpu(logs):\n    vals = [json.loads(line)['cpu'] for line in logs]\n    return sum(vals) / len(vals) if vals else 0.0\n```",
      category: "Python",
      difficulty: "Medium",
    },
    {
      id: "sample-2",
      instruction: "Generate a SQL query to select model variant names and average benchmark scores.",
      output: "```sql\nSELECT variant_name, AVG(eval_score) AS avg_score\nFROM model_benchmarks\nGROUP BY variant_name;\n```",
      category: "SQL",
      difficulty: "Easy",
    },
  ]);

  const [mcpTools, setMcpTools] = useState<MCPToolDeclaration[]>(DEFAULT_MCP_PRESETS);

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
      setOllamaConnected(true);
    }
  };

  useEffect(() => {
    checkOllamaConnection();
  }, []);

  const hardwareFit = calculateHardwareCompatibility(
    selectedModel,
    {
      lora_r: 32,
      lora_alpha: 32,
      lora_dropout: 0,
      learning_rate: 0.0002,
      batch_size: 2,
      gradient_accumulation_steps: 4,
      epochs: 3,
      max_seq_length: 4096,
      optimizer: "adamw_8bit",
      weight_decay: 0.01,
      warmup_steps: 10,
      use_gradient_checkpointing: true,
      use_unsloth_fast_backprop: true,
      use_dora: true,
    },
    {
      quantization: "Q4_K_M",
      contextLength: 4096,
      temperature: 0.6,
      top_p: 0.9,
      systemPrompt: "You are an expert tool execution AI.",
      num_gpu_layers: 999,
    },
    {
      enabled: true,
      pruneMethod: "structured_layer",
      layerPruningRange: [16, 23],
      headsPrunePercentage: 20,
      vocabTrimTarget: 32000,
      healingLoraSteps: 100,
    },
    {
      enabled: true,
      method: "dare_ties",
      numExperts: 4,
      topK: 2,
      routerType: "softmax",
      expertSources: [],
    }
  );

  return (
    <div className="min-h-screen bg-[#09090b] text-[#e4e4e7] flex flex-col font-sans selection:bg-blue-600/30 selection:text-blue-200">
      {/* Top Navigation HUD */}
      <Header
        currentStep={currentStep}
        setCurrentStep={setCurrentStep}
        activeMode={activeMode}
        setActiveMode={setActiveMode}
        activeStudioTab={activeStudioTab}
        setActiveStudioTab={setActiveStudioTab}
        selectedModel={selectedModel}
        hardwareFit={hardwareFit}
        ollamaConnected={ollamaConnected}
        checkOllamaConnection={checkOllamaConnection}
        onOpenOllamaManager={() => setIsOllamaManagerOpen(true)}
        onOpenDishwasher={() => setIsDishwasherOpen(true)}
      />

      {/* Main Content Workspace */}
      <main className="flex-1 max-w-7xl w-full mx-auto p-4 sm:p-6 lg:p-8">
        {activeMode === "pipeline" ? (
          <EasyPipelineWizard
            currentStep={currentStep}
            setCurrentStep={setCurrentStep}
            selectedModel={selectedModel}
            setSelectedModel={setSelectedModel}
          />
        ) : (
          <div className="space-y-6">
            {/* Studio Module Switcher Content */}
            {activeStudioTab === "data_mcp" && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 border-b border-zinc-800 pb-3 font-mono text-xs">
                  <button
                    onClick={() => setDataSubTab("synthetic")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      dataSubTab === "synthetic" ? "bg-blue-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    🧪 Synthetic Dataset Studio
                  </button>
                  <button
                    onClick={() => setDataSubTab("mcp")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      dataSubTab === "mcp" ? "bg-blue-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    🛠️ MCP Function-Calling Harness
                  </button>
                  <button
                    onClick={() => setDataSubTab("distillation")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      dataSubTab === "distillation" ? "bg-blue-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    🎓 Teacher-Student Distillation
                  </button>
                </div>
                {dataSubTab === "synthetic" && (
                  <DatasetStudio
                    dataset={dataset}
                    setDataset={setDataset}
                    onProceed={() => setActiveStudioTab("advisor")}
                  />
                )}
                {dataSubTab === "mcp" && (
                  <MCPHarnessStudio
                    mcpTools={mcpTools}
                    setMcpTools={setMcpTools}
                    onProceed={() => setActiveStudioTab("arena")}
                  />
                )}
                {dataSubTab === "distillation" && (
                  <ModelDistillationStudio onProceed={() => setActiveStudioTab("advisor")} />
                )}
              </div>
            )}

            {activeStudioTab === "advisor" && (
              <div className="space-y-6">
                <TechniqueWorkshop
                  selectedModel={selectedModel}
                  onProceed={() => setActiveStudioTab("pruning")}
                />
                <TrainingSimulator selectedModel={selectedModel} />
              </div>
            )}

            {activeStudioTab === "pruning" && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 border-b border-zinc-800 pb-3 font-mono text-xs">
                  <button
                    onClick={() => setPruneSubTab("moe")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      pruneSubTab === "moe" ? "bg-indigo-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    ✂️ MoE Expert Router Profiler
                  </button>
                  <button
                    onClick={() => setPruneSubTab("layer")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      pruneSubTab === "layer" ? "bg-indigo-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    🔍 Structured Layer & Head Pruner
                  </button>
                </div>
                {pruneSubTab === "moe" ? (
                  <MoEStudio selectedModel={selectedModel} />
                ) : (
                  <PruningStudio selectedModel={selectedModel} />
                )}
              </div>
            )}

            {activeStudioTab === "deploy" && (
              <div className="space-y-4">
                <div className="flex items-center gap-2 border-b border-zinc-800 pb-3 font-mono text-xs">
                  <button
                    onClick={() => setDeploySubTab("gguf")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      deploySubTab === "gguf" ? "bg-emerald-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    📦 GGUF Quantization Studio
                  </button>
                  <button
                    onClick={() => setDeploySubTab("ollama")}
                    className={`px-3 py-1.5 rounded-lg cursor-pointer ${
                      deploySubTab === "ollama" ? "bg-emerald-600 text-white font-bold" : "bg-zinc-900 text-zinc-400"
                    }`}
                  >
                    🚀 Direct Ollama Registry Deployer
                  </button>
                </div>
                {deploySubTab === "gguf" ? (
                  <GGUFStudio selectedModel={selectedModel} />
                ) : (
                  <OllamaDeployer selectedModel={selectedModel} ollamaConnected={ollamaConnected} />
                )}
              </div>
            )}

            {activeStudioTab === "arena" && (
              <InteractiveArena
                selectedModel={selectedModel}
                mcpTools={mcpTools}
                ollamaConnected={ollamaConnected}
              />
            )}
          </div>
        )}
      </main>

      {/* Ollama Models & CLI Terminal Modal */}
      <OllamaTerminalModal
        isOpen={isOllamaManagerOpen}
        onClose={() => setIsOllamaManagerOpen(false)}
      />

      {/* Dishwasher Autopilot Factory Modal */}
      <DishwasherModal
        isOpen={isDishwasherOpen}
        onClose={() => setIsDishwasherOpen(false)}
        selectedModel={selectedModel}
      />

      {/* Persistent Footer */}
      <footer className="border-t border-[#27272a] bg-[#0c0c0e] py-3.5 px-6 text-center text-xs text-zinc-500">
        <div className="max-w-7xl mx-auto flex flex-wrap items-center justify-between gap-2">
          <div className="font-mono text-[11px]">
            Ollama Personal Trainer • Target Hardware: NVIDIA RTX 4080 Super (16GB VRAM) + 64GB System RAM
          </div>
          <div className="flex items-center gap-3 text-zinc-400 font-mono text-[11px]">
            <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">CUDA 12.4+</span>
            <span className="px-2 py-0.5 rounded bg-zinc-900 border border-zinc-800 text-zinc-300">Unsloth Triton</span>
            <span className="px-2 py-0.5 rounded bg-emerald-950/40 border border-emerald-500/30 text-emerald-400">Ollama GGUF Ready</span>
          </div>
        </div>
      </footer>
    </div>
  );
}

