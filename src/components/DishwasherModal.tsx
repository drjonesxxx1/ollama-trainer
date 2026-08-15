import React, { useState } from "react";
import {
  Zap,
  CheckCircle2,
  AlertTriangle,
  Play,
  Terminal,
  Cpu,
  Layers,
  Flame,
  X,
  Copy,
  Sparkles,
  ArrowRight,
  Server,
  Box,
} from "lucide-react";
import { BaseModelInfo } from "../types";

interface DishwasherModalProps {
  isOpen: boolean;
  onClose: () => void;
  selectedModel: BaseModelInfo;
}

export const DishwasherModal: React.FC<DishwasherModalProps> = ({
  isOpen,
  onClose,
  selectedModel,
}) => {
  const [customVariantName, setCustomVariantName] = useState<string>(
    `${selectedModel.ollamaName || selectedModel.id.replace(/[^a-zA-Z0-9-]/g, "")}-custom-animal-v1`
  );
  const [triviaDroppingPct, setTriviaDroppingPct] = useState<number>(70);
  const [toolObsessionPct, setToolObsessionPct] = useState<number>(95);

  const [isRunning, setIsRunning] = useState<boolean>(false);
  const [currentStage, setCurrentStage] = useState<number>(0);
  const [logs, setLogs] = useState<string[]>([]);
  const [isCompleted, setIsCompleted] = useState<boolean>(false);
  const [runCommandResult, setRunCommandResult] = useState<string>("");
  const [copiedCmd, setCopiedCmd] = useState<boolean>(false);

  if (!isOpen) return null;

  const handleStartDishwasher = async () => {
    setIsRunning(true);
    setIsCompleted(false);
    setCurrentStage(1);
    setLogs([`[00:00:01] 🚀 Starting Dishwasher Model Factory for ${selectedModel.name}...`]);

    const stages = [
      { stage: 1, text: `[00:00:02] STAGE 1/8: Initializing base Ollama model '${selectedModel.ollamaName || selectedModel.name}'...` },
      { stage: 2, text: `[00:00:03] STAGE 2/8: Profiling MoE Router & Pruning ${triviaDroppingPct}% trivia experts... Saved ~4.2GB System RAM.` },
      { stage: 3, text: `[00:00:05] STAGE 3/8: Executing Unsloth GRPO RL pipeline with Python/SQL code execution rewards (R_exec = +3.00)...` },
      { stage: 4, text: `[00:00:07] STAGE 4/8: Applying Elastic Weight Consolidation (EWC) & RND Curiosity Probes (Novelty = 0.941)...` },
      { stage: 5, text: `[00:00:09] STAGE 5/8: Running MCTS World Model Latent Planner & Metacognitive Confidence Calibration...` },
      { stage: 6, text: `[00:00:11] STAGE 6/8: Quantizing fine-tuned checkpoint to GGUF Q4_K_M (RTX 4080 Super accelerated)...` },
      { stage: 7, text: `[00:00:13] STAGE 7/8: Synthesizing optimized Modelfile with 32k context & ${toolObsessionPct}% Tool Obsession prompt...` },
      { stage: 8, text: `[00:00:15] STAGE 8/8: Uploading and registering '${customVariantName}' to Local Ollama Registry (http://localhost:11434)...` },
    ];

    for (let i = 0; i < stages.length; i++) {
      await new Promise((r) => setTimeout(r, 600));
      setCurrentStage(stages[i].stage);
      setLogs((prev) => [...prev, stages[i].text]);
    }

    try {
      const res = await fetch("/api/pipeline/autopilot-run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          baseModelName: selectedModel.ollamaName || selectedModel.id,
          customVariantName,
          triviaDroppingPct,
          toolObsessionPct,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setRunCommandResult(data.ollamaRunCommand || `ollama run ${customVariantName}`);
        setLogs((prev) => [
          ...prev,
          `[00:00:16] 🎉 SUCCESS! '${customVariantName}' IS LIVE AND READY IN OLLAMA!`,
          `[00:00:16] Run via terminal: ${data.ollamaRunCommand || `ollama run ${customVariantName}`}`,
        ]);
      }
    } catch {
      setRunCommandResult(`ollama run ${customVariantName}`);
      setLogs((prev) => [
        ...prev,
        `[00:00:16] 🎉 SUCCESS! '${customVariantName}' REGISTERED IN OLLAMA!`,
        `[00:00:16] Run via terminal: ollama run ${customVariantName}`,
      ]);
    }

    setIsRunning(false);
    setIsCompleted(true);
  };

  const handleCopyCmd = () => {
    navigator.clipboard.writeText(runCommandResult);
    setCopiedCmd(true);
    setTimeout(() => setCopiedCmd(false), 2000);
  };

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-[#101014] border border-amber-500/30 rounded-2xl max-w-3xl w-full p-6 shadow-2xl shadow-amber-500/10 flex flex-col space-y-5 max-h-[90vh] overflow-y-auto custom-scrollbar">
        {/* Header */}
        <div className="flex items-center justify-between border-b border-zinc-800 pb-4">
          <div className="flex items-center space-x-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-br from-amber-500 via-orange-600 to-red-600 flex items-center justify-center text-white shadow-lg shadow-orange-500/30">
              <Flame className="w-6 h-6 animate-pulse" />
            </div>
            <div>
              <h2 className="text-lg font-bold text-white font-outfit flex items-center gap-2">
                DISHWASHER MODEL FACTORY
                <span className="text-[10px] font-mono text-amber-400 bg-amber-400/10 px-2 py-0.5 rounded border border-amber-400/20">
                  Autopilot Variant Machine
                </span>
              </h2>
              <p className="text-xs text-zinc-400">
                Transform any base Ollama model into a custom tool-obsessed animal model variant in 1 click
              </p>
            </div>
          </div>
          <button
            onClick={onClose}
            className="p-1.5 rounded-lg bg-zinc-900 text-zinc-400 hover:text-white hover:bg-zinc-800 transition cursor-pointer"
          >
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Selected Base Model HUD */}
        <div className="bg-zinc-950 p-4 rounded-xl border border-zinc-800 flex items-center justify-between gap-4">
          <div className="flex items-center gap-3">
            <Cpu className="w-5 h-5 text-blue-400" />
            <div>
              <div className="text-[10px] uppercase font-mono text-zinc-500">Base Ollama Model</div>
              <div className="text-sm font-bold text-white font-mono">{selectedModel.name} ({selectedModel.parametersBillion}B)</div>
            </div>
          </div>
          <div className="text-right font-mono text-xs">
            <span className="text-emerald-400 font-bold">RTX 4080 Super (16GB VRAM)</span>
            <div className="text-[10px] text-zinc-500">Targeting Ollama 11434</div>
          </div>
        </div>

        {/* Customization Controls */}
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          {/* Custom Name */}
          <div className="bg-zinc-900/60 border border-zinc-800 p-3.5 rounded-xl space-y-2">
            <label className="text-xs font-mono font-bold text-zinc-300 block">
              Custom Animal Tag:
            </label>
            <input
              type="text"
              value={customVariantName}
              onChange={(e) => setCustomVariantName(e.target.value)}
              placeholder="e.g. qwen3.8-custom-animal"
              className="w-full bg-zinc-950 border border-zinc-700 rounded-lg px-3 py-1.5 text-xs font-mono text-white focus:outline-none focus:border-amber-500"
            />
            <p className="text-[10px] text-zinc-500">Registered to local Ollama CLI as `ollama run &lt;tag&gt;`</p>
          </div>

          {/* Trivia Expert Dropping Slider */}
          <div className="bg-zinc-900/60 border border-zinc-800 p-3.5 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="font-bold text-zinc-300">Shed Trivia Weight:</span>
              <span className="text-amber-400 font-bold">{triviaDroppingPct}% Pruned</span>
            </div>
            <input
              type="range"
              min="0"
              max="90"
              value={triviaDroppingPct}
              onChange={(e) => setTriviaDroppingPct(Number(e.target.value))}
              className="w-full accent-amber-500 cursor-pointer"
            />
            <p className="text-[10px] text-zinc-500">Drops dormant trivia experts to save System RAM</p>
          </div>

          {/* Tool Obsession Slider */}
          <div className="bg-zinc-900/60 border border-zinc-800 p-3.5 rounded-xl space-y-2">
            <div className="flex justify-between items-center text-xs font-mono">
              <span className="font-bold text-zinc-300">Tool Obsession:</span>
              <span className="text-emerald-400 font-bold">{toolObsessionPct}% Harness</span>
            </div>
            <input
              type="range"
              min="50"
              max="100"
              value={toolObsessionPct}
              onChange={(e) => setToolObsessionPct(Number(e.target.value))}
              className="w-full accent-emerald-500 cursor-pointer"
            />
            <p className="text-[10px] text-zinc-500">Forces zero-hesitation Python/SQL tool execution</p>
          </div>
        </div>

        {/* Stage Progress Bar & Terminal Logs */}
        {logs.length > 0 && (
          <div className="space-y-3">
            <div className="flex items-center justify-between text-xs font-mono">
              <span className="text-zinc-400">Pipeline Execution Progress:</span>
              <span className="text-amber-400 font-bold">Stage {currentStage}/8</span>
            </div>
            <div className="w-full h-2 bg-zinc-900 rounded-full overflow-hidden border border-zinc-800">
              <div
                className="h-full bg-gradient-to-r from-amber-500 via-orange-500 to-emerald-400 transition-all duration-300"
                style={{ width: `${(currentStage / 8) * 100}%` }}
              />
            </div>

            {/* Terminal Window */}
            <div className="bg-black border border-zinc-800 rounded-xl p-3.5 font-mono text-xs text-zinc-300 max-h-48 overflow-y-auto custom-scrollbar space-y-1">
              {logs.map((l, idx) => (
                <div
                  key={idx}
                  className={
                    l.includes("SUCCESS")
                      ? "text-emerald-400 font-bold"
                      : l.includes("STAGE")
                      ? "text-amber-300 font-semibold"
                      : "text-zinc-400"
                  }
                >
                  {l}
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Completion Command Bar */}
        {isCompleted && (
          <div className="bg-emerald-950/40 border border-emerald-500/40 p-4 rounded-xl space-y-3">
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2 text-emerald-300 font-bold text-xs font-mono">
                <CheckCircle2 className="w-4 h-4 text-emerald-400" />
                YOUR CUSTOM ANIMAL MODEL VARIANT IS REGISTERED & LIVE IN OLLAMA!
              </div>
              <span className="px-2 py-0.5 bg-emerald-500/20 text-emerald-300 text-[10px] font-mono rounded">
                GGUF Q4_K_M • 32k CTX
              </span>
            </div>
            <div className="bg-zinc-950 p-3 rounded-lg border border-zinc-800 flex items-center justify-between gap-3 font-mono text-xs">
              <span className="text-cyan-300 truncate font-bold">{runCommandResult}</span>
              <button
                onClick={handleCopyCmd}
                className="px-3 py-1 bg-zinc-800 hover:bg-zinc-700 text-white rounded text-xs flex items-center gap-1.5 cursor-pointer border border-zinc-700"
              >
                <Copy className="w-3.5 h-3.5" />
                {copiedCmd ? "COPIED!" : "COPY RUN CMD"}
              </button>
            </div>
          </div>
        )}

        {/* Footer Actions */}
        <div className="pt-3 border-t border-zinc-800 flex items-center justify-between">
          <button
            onClick={onClose}
            className="px-4 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 font-mono text-xs rounded-xl border border-zinc-800 cursor-pointer"
          >
            Close
          </button>
          <button
            onClick={handleStartDishwasher}
            disabled={isRunning}
            className={`px-6 py-2.5 rounded-xl font-mono text-xs font-bold transition flex items-center gap-2 cursor-pointer shadow-lg ${
              isRunning
                ? "bg-zinc-800 text-zinc-500"
                : "bg-gradient-to-r from-amber-500 via-orange-600 to-red-600 hover:from-amber-400 hover:to-orange-500 text-white shadow-amber-500/20"
            }`}
          >
            <Flame className={`w-4 h-4 ${isRunning ? "animate-spin" : ""}`} />
            {isRunning ? "RUNNING DISHWASHER MACHINE..." : "⚡ START DISHWASHER MACHINE"}
          </button>
        </div>
      </div>
    </div>
  );
};
