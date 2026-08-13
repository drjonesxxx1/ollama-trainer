import React, { useState } from "react";
import {
  Cpu,
  Zap,
  Server,
  Sparkles,
  Wrench,
  Database,
  Sliders,
  Scissors,
  Network,
  Binary,
  PlayCircle,
  UploadCloud,
  MessageSquare,
  CheckCircle2,
  RefreshCw,
} from "lucide-react";
import { ActiveTab, BaseModelInfo } from "../types";
import { VRAMCalculationResult } from "../utils/hardwareCalculator";

interface HeaderProps {
  activeTab: ActiveTab;
  setActiveTab: (tab: ActiveTab) => void;
  selectedModel: BaseModelInfo;
  hardwareFit?: VRAMCalculationResult;
  ollamaConnected: boolean;
  checkOllamaConnection: () => void;
  vramUsedPercent?: number;
  totalVramUsedGb?: number;
}

export const Header: React.FC<HeaderProps> = ({
  activeTab,
  setActiveTab,
  selectedModel,
  hardwareFit,
  ollamaConnected,
  checkOllamaConnection,
  vramUsedPercent = hardwareFit?.utilizationPercent || 68,
  totalVramUsedGb = hardwareFit?.totalTrainingVramGb || 10.8,
}) => {
  const [checkingOllama, setCheckingOllama] = useState(false);

  const handleRefreshOllama = async () => {
    setCheckingOllama(true);
    await checkOllamaConnection();
    setTimeout(() => setCheckingOllama(false), 500);
  };

  const navItems: { id: ActiveTab; label: string; icon: React.ReactNode; badge?: string }[] = [
    { id: "models", label: "1. Base Model", icon: <Cpu className="w-3.5 h-3.5" /> },
    { id: "techniques", label: "2. Techniques", icon: <Sliders className="w-3.5 h-3.5" />, badge: "20+" },
    { id: "dataset", label: "3. Dataset & AI", icon: <Database className="w-3.5 h-3.5" /> },
    { id: "mcp_harness", label: "4. MCP Plugins", icon: <Wrench className="w-3.5 h-3.5" />, badge: "MCP" },
    { id: "distillation", label: "5. Distillation", icon: <Sparkles className="w-3.5 h-3.5" /> },
    { id: "pruning", label: "6. Slim / Prune", icon: <Scissors className="w-3.5 h-3.5" /> },
    { id: "moe_merge", label: "7. MoE & Merge", icon: <Network className="w-3.5 h-3.5" /> },
    { id: "gguf", label: "8. GGUF Matrix", icon: <Binary className="w-3.5 h-3.5" /> },
    { id: "train", label: "9. Training Run", icon: <PlayCircle className="w-3.5 h-3.5" /> },
    { id: "deploy", label: "10. Ollama Export", icon: <UploadCloud className="w-3.5 h-3.5" />, badge: "4080" },
    { id: "arena", label: "Arena Playground", icon: <MessageSquare className="w-3.5 h-3.5" /> },
  ];

  return (
    <header className="bg-[#121214] border-b border-[#27272a] text-[#e4e4e7] sticky top-0 z-50">
      {/* Top Meta Bar */}
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-3.5 flex flex-wrap items-center justify-between gap-4">
        {/* Brand */}
        <div className="flex items-center space-x-3">
          <div className="w-8 h-8 bg-blue-600 rounded flex items-center justify-center font-bold text-white shadow-sm shadow-blue-600/30 text-base">
            Ω
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <h1 className="font-semibold text-lg text-[#f4f4f5] tracking-tight">
                Ollama Unsloth Studio
              </h1>
              <span className="text-[11px] font-normal text-blue-400 bg-blue-400/10 px-2 py-0.5 rounded border border-blue-400/20 font-mono">
                v2.4 Pro
              </span>
            </div>
            <p className="text-[11px] text-zinc-400">
              Unsloth & GGUF Pipeline for Windows RTX 4080 Super • Ollama Native
            </p>
          </div>
        </div>

        {/* Hardware Status HUD */}
        <div className="flex items-center flex-wrap gap-3 text-xs">
          {/* Target GPU Badge */}
          <div className="flex flex-col items-end px-3 py-1 bg-zinc-950/80 rounded border border-zinc-800">
            <span className="text-[10px] uppercase tracking-wider text-zinc-500 font-mono">Hardware Target</span>
            <span className="text-xs font-mono text-emerald-400 font-medium">RTX 4080 SUPER • 16GB VRAM</span>
          </div>

          {/* VRAM Meter */}
          <div className="px-3 py-1.5 rounded bg-zinc-950/80 border border-zinc-800 min-w-[130px]">
            <div className="flex justify-between items-center text-[10px] uppercase tracking-wider font-mono mb-1">
              <span className="text-zinc-500">VRAM Load</span>
              <span className={vramUsedPercent > 95 ? "text-rose-400 font-bold" : "text-emerald-400"}>
                {totalVramUsedGb} / 16 GB
              </span>
            </div>
            <div className="w-full h-1 bg-zinc-800 rounded-full overflow-hidden">
              <div
                className={`h-full rounded-full transition-all duration-300 ${
                  vramUsedPercent > 95
                    ? "bg-rose-500"
                    : vramUsedPercent > 80
                    ? "bg-amber-400"
                    : "bg-blue-500"
                }`}
                style={{ width: `${Math.min(vramUsedPercent, 100)}%` }}
              />
            </div>
          </div>

          {/* Ollama Local Status */}
          <button
            onClick={handleRefreshOllama}
            title="Click to re-check local Ollama service (http://localhost:11434)"
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded border transition-colors cursor-pointer text-xs ${
              ollamaConnected
                ? "bg-emerald-950/30 border-emerald-500/20 text-emerald-300 hover:bg-emerald-900/40"
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
              Ollama: {ollamaConnected ? "Connected" : "11434"}
            </span>
          </button>
        </div>
      </div>

      {/* Navigation Pipeline Tabs */}
      <div className="bg-[#0c0c0e] border-t border-[#27272a] overflow-x-auto scrollbar-none">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-1.5">
          <nav className="flex space-x-1.5">
            {navItems.map((item) => {
              const isActive = activeTab === item.id;
              return (
                <button
                  key={item.id}
                  id={`tab-${item.id}`}
                  onClick={() => setActiveTab(item.id)}
                  className={`flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium whitespace-nowrap transition-all cursor-pointer ${
                    isActive
                      ? "bg-zinc-800/80 text-blue-400 border border-blue-500/30 shadow-sm font-semibold"
                      : "text-zinc-400 hover:text-zinc-200 hover:bg-zinc-800/40 border border-transparent"
                  }`}
                >
                  {item.icon}
                  <span>{item.label}</span>
                  {item.badge && (
                    <span
                      className={`px-1.5 py-0.2 rounded text-[10px] font-bold font-mono ${
                        isActive
                          ? "bg-blue-600 text-white"
                          : "bg-zinc-900 text-zinc-400 border border-zinc-800"
                      }`}
                    >
                      {item.badge}
                    </span>
                  )}
                </button>
              );
            })}
          </nav>
        </div>
      </div>
    </header>
  );
};
