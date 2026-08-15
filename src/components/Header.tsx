import React, { useState } from "react";
import {
  Cpu,
  Zap,
  Server,
  Sliders,
  PlayCircle,
  MessageSquare,
  CheckCircle2,
  RefreshCw,
  Terminal as TerminalIcon,
  UploadCloud,
  Brain,
  Compass,
  Globe,
  Eye,
  Database,
  Wrench,
  Scissors,
  Box,
  Swords,
  Layers,
  Flame,
} from "lucide-react";
import { BaseModelInfo } from "../types";
import { VRAMCalculationResult } from "../utils/hardwareCalculator";

interface HeaderProps {
  currentStep: number;
  setCurrentStep: (step: number) => void;
  activeMode: "pipeline" | "studio";
  setActiveMode: (mode: "pipeline" | "studio") => void;
  activeStudioTab: string;
  setActiveStudioTab: (tab: string) => void;
  selectedModel: BaseModelInfo;
  hardwareFit?: VRAMCalculationResult;
  ollamaConnected: boolean;
  checkOllamaConnection: () => void;
  onOpenOllamaManager: () => void;
  onOpenDishwasher: () => void;
}

export const Header: React.FC<HeaderProps> = ({
  currentStep,
  setCurrentStep,
  activeMode,
  setActiveMode,
  activeStudioTab,
  setActiveStudioTab,
  selectedModel,
  hardwareFit,
  ollamaConnected,
  checkOllamaConnection,
  onOpenOllamaManager,
  onOpenDishwasher,
}) => {
  const [checkingOllama, setCheckingOllama] = useState(false);

  const handleRefreshOllama = async () => {
    setCheckingOllama(true);
    await checkOllamaConnection();
    setTimeout(() => setCheckingOllama(false), 500);
  };

  const pipelineSteps = [
    { id: 1, label: "1. Pick Model", icon: <Cpu className="w-4 h-4" />, desc: "Select base model" },
    { id: 2, label: "2. Human Knobs", icon: <Sliders className="w-4 h-4" />, desc: "Shed weight & tool obsession" },
    { id: 3, label: "3. GPU Pipeline", icon: <PlayCircle className="w-4 h-4" />, desc: "Unsloth GRPO training" },
    { id: 4, label: "4. Test Drive", icon: <MessageSquare className="w-4 h-4" />, desc: "Interactive chat" },
    { id: 5, label: "5. Plasticity", icon: <Brain className="w-4 h-4" />, desc: "EWC online learning" },
    { id: 6, label: "6. Curiosity", icon: <Compass className="w-4 h-4" />, desc: "RND exploration" },
    { id: 7, label: "7. World Model", icon: <Globe className="w-4 h-4" />, desc: "MCTS planner" },
    { id: 8, label: "8. Metacognition", icon: <Eye className="w-4 h-4" />, desc: "Self-refinement" },
  ];

  const studioTabs = [
    { id: "data_mcp", label: "Dataset & MCP Studio", icon: <Database className="w-4 h-4" />, desc: "Synthetic Data & MCP Tool Calling" },
    { id: "advisor", label: "Training Advisor", icon: <Wrench className="w-4 h-4" />, desc: "Unsloth Hyperparameter & VRAM Simulator" },
    { id: "pruning", label: "MoE & Layer Pruner", icon: <Scissors className="w-4 h-4" />, desc: "Expert Dropping & Head Trimming" },
    { id: "deploy", label: "GGUF & Ollama Deploy", icon: <Box className="w-4 h-4" />, desc: "Quantize & Register GGUF Modelfiles" },
    { id: "arena", label: "Variant Battle Arena", icon: <Swords className="w-4 h-4" />, desc: "Side-by-side Model Benchmark Arena" },
  ];

  return (
    <header className="bg-[#0c0c0e] border-b border-[#27272a] text-[#e4e4e7] sticky top-0 z-50 shadow-lg">
      {/* Top Meta Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3 flex flex-wrap items-center justify-between gap-4">
        {/* Brand & Mode Switcher */}
        <div className="flex items-center space-x-4">
          <div className="w-9 h-9 bg-gradient-to-br from-blue-600 via-indigo-600 to-emerald-500 rounded-lg flex items-center justify-center font-bold text-white shadow-md shadow-blue-500/20 text-lg font-mono">
            ⚡
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-bold text-lg text-[#f4f4f5] tracking-tight">
                Ollama Personal Trainer
              </h1>
              <span className="text-[10px] font-mono text-emerald-400 bg-emerald-400/10 px-2 py-0.5 rounded border border-emerald-400/20">
                v2.0 Orchestrator
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Human-Friendly Model Pruning & Tool-Obsessed Fine-Tuner for NVIDIA GPUs
            </p>
          </div>

          {/* Mode Switcher Toggle */}
          <div className="flex items-center bg-zinc-950 p-1 rounded-xl border border-zinc-800 ml-4 font-mono text-xs">
            <button
              onClick={() => setActiveMode("pipeline")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition cursor-pointer font-semibold ${
                activeMode === "pipeline"
                  ? "bg-blue-600 text-white shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Zap className="w-3.5 h-3.5" /> 8-Step Pipeline
            </button>
            <button
              onClick={() => setActiveMode("studio")}
              className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg transition cursor-pointer font-semibold ${
                activeMode === "studio"
                  ? "bg-indigo-600 text-white shadow"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              <Layers className="w-3.5 h-3.5" /> Studio Suite
            </button>
          </div>
        </div>

        {/* Hardware Status HUD & Dishwasher Machine Launcher */}
        <div className="flex items-center flex-wrap gap-3 text-xs">
          {/* Dishwasher Autopilot Factory Launcher */}
          <button
            onClick={onOpenDishwasher}
            className="flex items-center gap-2 px-3 py-1.5 bg-gradient-to-r from-amber-500 via-orange-600 to-red-600 hover:from-amber-400 hover:to-orange-500 text-white font-bold text-xs rounded-lg transition shadow-md shadow-orange-600/30 font-mono cursor-pointer"
          >
            <Flame className="w-3.5 h-3.5 text-yellow-300 animate-pulse" />
            <span>⚡ DISHWASHER FACTORY</span>
          </button>

          {/* Target GPU Badge */}
          <div className="flex items-center gap-2 px-3 py-1.5 bg-zinc-950 rounded-lg border border-zinc-800">
            <Zap className="w-3.5 h-3.5 text-amber-400" />
            <div>
              <div className="text-[9px] uppercase tracking-wider text-zinc-500 font-mono">NVIDIA Target</div>
              <div className="text-xs font-mono text-emerald-400 font-semibold">RTX 4080 SUPER • 16GB VRAM + 64GB RAM</div>
            </div>
          </div>

          {/* Ollama Local Status Button */}
          <button
            onClick={handleRefreshOllama}
            title="Click to re-check local Ollama service (http://localhost:11434)"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded-lg border transition-colors cursor-pointer text-xs ${
              ollamaConnected
                ? "bg-emerald-950/40 border-emerald-500/30 text-emerald-300 hover:bg-emerald-900/50"
                : "bg-zinc-900 border-zinc-800 text-zinc-300 hover:bg-zinc-800"
            }`}
          >
            {checkingOllama ? (
              <RefreshCw className="w-3.5 h-3.5 animate-spin text-zinc-400" />
            ) : ollamaConnected ? (
              <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400" />
            ) : (
              <Server className="w-3.5 h-3.5 text-zinc-400" />
            )}
            <span className="font-mono text-[11px]">
              Ollama: {ollamaConnected ? "Ready (11434)" : "Disconnected"}
            </span>
          </button>

          {/* Ollama Terminal & Model Manager Launcher */}
          <button
            onClick={onOpenOllamaManager}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-600 hover:bg-blue-500 text-white font-bold text-xs rounded-lg transition shadow-md shadow-blue-600/20 font-mono cursor-pointer"
          >
            <TerminalIcon className="w-3.5 h-3.5 text-cyan-300" />
            <span>Ollama Terminal & Models</span>
          </button>
        </div>
      </div>

      {/* Sub-Navigation Bar (Pipeline Steps vs Studio Tabs) */}
      <div className="bg-[#08080a] border-t border-[#27272a] py-2">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          {activeMode === "pipeline" ? (
            <div className="flex gap-2 overflow-x-auto pb-1 custom-scrollbar">
              {pipelineSteps.map((step) => {
                const isActive = currentStep === step.id;
                const isCompleted = currentStep > step.id;
                return (
                  <button
                    key={step.id}
                    onClick={() => setCurrentStep(step.id)}
                    className={`flex items-center gap-2 px-3 py-2 rounded-xl text-xs font-medium transition-all text-left cursor-pointer border shrink-0 ${
                      isActive
                        ? "bg-blue-600/15 border-blue-500/50 text-blue-300 shadow-md shadow-blue-500/10 font-semibold"
                        : isCompleted
                        ? "bg-emerald-950/20 border-emerald-500/30 text-emerald-400"
                        : "bg-zinc-900/40 border-zinc-800/80 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
                    }`}
                  >
                    <div
                      className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 ${
                        isActive
                          ? "bg-blue-600 text-white"
                          : isCompleted
                          ? "bg-emerald-500 text-black font-extrabold"
                          : "bg-zinc-800 text-zinc-400"
                      }`}
                    >
                      {isCompleted ? "✓" : step.icon}
                    </div>
                    <div className="min-w-0">
                      <div className="truncate font-semibold">{step.label}</div>
                      <div className="text-[10px] text-zinc-500 truncate hidden sm:block">{step.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          ) : (
            <div className="flex gap-2 overflow-x-auto pb-1 custom-scrollbar">
              {studioTabs.map((tab) => {
                const isActive = activeStudioTab === tab.id;
                return (
                  <button
                    key={tab.id}
                    onClick={() => setActiveStudioTab(tab.id)}
                    className={`flex items-center gap-2.5 px-4 py-2 rounded-xl text-xs font-medium transition-all text-left cursor-pointer border shrink-0 ${
                      isActive
                        ? "bg-indigo-600/20 border-indigo-500/60 text-indigo-300 shadow-md shadow-indigo-500/10 font-semibold"
                        : "bg-zinc-900/40 border-zinc-800/80 text-zinc-400 hover:bg-zinc-800/60 hover:text-zinc-200"
                    }`}
                  >
                    <div
                      className={`w-7 h-7 rounded-lg flex items-center justify-center font-bold text-xs shrink-0 ${
                        isActive ? "bg-indigo-600 text-white" : "bg-zinc-800 text-zinc-400"
                      }`}
                    >
                      {tab.icon}
                    </div>
                    <div className="min-w-0">
                      <div className="truncate font-semibold">{tab.label}</div>
                      <div className="text-[10px] text-zinc-500 truncate hidden sm:block">{tab.desc}</div>
                    </div>
                  </button>
                );
              })}
            </div>
          )}
        </div>
      </div>
    </header>
  );
};
