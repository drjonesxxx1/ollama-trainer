import React, { useState } from "react";
import {
  Sliders,
  Sparkles,
  Zap,
  Check,
  Code,
  RefreshCw,
  ArrowRight,
} from "lucide-react";
import { BaseModelInfo, TrainingHyperparameters } from "../types";
import { SOTA_TECHNIQUES } from "../data/techniques";

interface TechniqueWorkshopProps {
  selectedModel: BaseModelInfo;
  hyperparameters: TrainingHyperparameters;
  setHyperparameters: React.Dispatch<React.SetStateAction<TrainingHyperparameters>>;
  onProceed: () => void;
}

export const TechniqueWorkshop: React.FC<TechniqueWorkshopProps> = ({
  selectedModel,
  hyperparameters,
  setHyperparameters,
  onProceed,
}) => {
  const [selectedTechniqueId, setSelectedTechniqueId] = useState<string>("qlora");
  const [aiOptimizing, setAiOptimizing] = useState(false);
  const [aiAdvisorResult, setAiAdvisorResult] = useState<any>(null);

  const activeTechnique = SOTA_TECHNIQUES.find((t) => t.id === selectedTechniqueId) || SOTA_TECHNIQUES[0];

  const handleConsultAIAdvisor = async () => {
    setAiOptimizing(true);
    try {
      const res = await fetch("/api/advisor/optimize-config", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modelName: selectedModel.name,
          targetHardware: "NVIDIA RTX 4080 Super",
          vramGb: 16,
          datasetSize: 1500,
          targetTask: "High precision instruction following & MCP Tool Calling",
          selectedTechniques: [selectedTechniqueId],
        }),
      });
      const data = await res.json();
      if (data.success && data.config) {
        setAiAdvisorResult(data.config);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setAiOptimizing(false);
    }
  };

  const applyAIRecommendation = () => {
    if (!aiAdvisorResult) return;
    setHyperparameters((prev) => ({
      ...prev,
      lora_r: aiAdvisorResult.recommendedLoRA_r || prev.lora_r,
      lora_alpha: aiAdvisorResult.recommendedLoRA_alpha || prev.lora_alpha,
      batch_size: aiAdvisorResult.batchSize || prev.batch_size,
      gradient_accumulation_steps: aiAdvisorResult.gradAccumSteps || prev.gradient_accumulation_steps,
      learning_rate: parseFloat(aiAdvisorResult.learningRate) || prev.learning_rate,
    }));
  };

  return (
    <div className="space-y-6">
      {/* AI Training Optimization Advisor Banner */}
      <div className="bg-[#18181b]/70 border border-blue-500/25 rounded-xl p-5 relative overflow-hidden">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="max-w-2xl">
            <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[11px] font-mono text-blue-400 bg-blue-400/10 border border-blue-400/20 mb-2">
              <Sparkles className="w-3.5 h-3.5 text-blue-400" /> GEMINI SOTA ADVISOR
            </div>
            <h3 className="text-base font-semibold text-[#f4f4f5]">
              AI Hyperparameter Auto-Tuner for RTX 4080 Super
            </h3>
            <p className="text-xs text-zinc-400 mt-1">
              Automatically calculate optimal LoRA rank, alpha, micro-batching, and learning rate for {selectedModel.name} on 16GB VRAM.
            </p>
          </div>

          <div className="flex items-center gap-2">
            <button
              id="btn-consult-ai-advisor"
              onClick={handleConsultAIAdvisor}
              disabled={aiOptimizing}
              className="flex items-center gap-2 px-3.5 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer disabled:opacity-50"
            >
              {aiOptimizing ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Computing Optimal Strategy...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Auto-Tune Strategy</span>
                </>
              )}
            </button>

            {aiAdvisorResult && (
              <button
                onClick={applyAIRecommendation}
                className="flex items-center gap-1.5 px-3 py-2 rounded text-xs font-medium bg-emerald-950/40 hover:bg-emerald-900/40 text-emerald-400 border border-emerald-500/30 cursor-pointer"
              >
                <Check className="w-3.5 h-3.5" /> Apply Recs
              </button>
            )}
          </div>
        </div>

        {/* AI Recommendations Output */}
        {aiAdvisorResult && (
          <div className="mt-4 pt-4 border-t border-[#27272a] grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
            <div className="bg-black/30 p-3 rounded border border-zinc-800">
              <div className="text-[10px] uppercase font-mono text-zinc-500">Recommended LoRA</div>
              <div className="font-mono font-medium text-zinc-200 text-xs mt-0.5">
                r={aiAdvisorResult.recommendedLoRA_r}, α={aiAdvisorResult.recommendedLoRA_alpha}
              </div>
            </div>
            <div className="bg-black/30 p-3 rounded border border-zinc-800">
              <div className="text-[10px] uppercase font-mono text-zinc-500">Micro-Batch / GradAccum</div>
              <div className="font-mono font-medium text-zinc-200 text-xs mt-0.5">
                {aiAdvisorResult.batchSize} / {aiAdvisorResult.gradAccumSteps} steps
              </div>
            </div>
            <div className="bg-black/30 p-3 rounded border border-zinc-800">
              <div className="text-[10px] uppercase font-mono text-zinc-500">Estimated Train VRAM</div>
              <div className="font-mono font-medium text-emerald-400 text-xs mt-0.5">
                {aiAdvisorResult.trainingVramEstimateGb} GB (Fits 16GB)
              </div>
            </div>
            <div className="bg-black/30 p-3 rounded border border-zinc-800">
              <div className="text-[10px] uppercase font-mono text-zinc-500">Fit Confidence</div>
              <div className="font-mono font-medium text-blue-400 text-xs mt-0.5">
                {aiAdvisorResult.fitProbabilityPercent}% Perfect Fit
              </div>
            </div>
          </div>
        )}
      </div>

      {/* Main Grid: Techniques Sidebar + Detailed Configuration */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Technique Selector */}
        <div className="space-y-3">
          <div className="text-[10px] font-mono font-bold text-zinc-500 uppercase px-1 flex items-center gap-1.5">
            <Zap className="w-3.5 h-3.5 text-blue-400" /> SOTA TECHNIQUES ({SOTA_TECHNIQUES.length})
          </div>
          <div className="space-y-1.5 max-h-[600px] overflow-y-auto pr-1">
            {SOTA_TECHNIQUES.map((tech) => {
              const isSelected = selectedTechniqueId === tech.id;
              return (
                <div
                  key={tech.id}
                  onClick={() => setSelectedTechniqueId(tech.id)}
                  className={`p-3 rounded-lg border transition-all cursor-pointer ${
                    isSelected
                      ? "bg-[#18181b] border-blue-500/40 text-blue-400 ring-1 ring-blue-500/20"
                      : "bg-[#121214] border-[#27272a] hover:border-zinc-700 hover:bg-[#18181b]/40 text-zinc-300"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-medium text-xs text-[#f4f4f5]">{tech.name}</span>
                    <span className="text-[10px] font-mono text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded border border-blue-400/20">
                      {tech.category}
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-400 mt-1 line-clamp-2">{tech.tagline}</p>
                </div>
              );
            })}
          </div>
        </div>

        {/* Right: Technique Deep Dive & Hyperparameter Controls */}
        <div className="lg:col-span-2 space-y-6">
          {/* Active Technique Overview Card */}
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5">
            <div className="flex flex-wrap items-center justify-between gap-2 mb-3">
              <div>
                <span className="text-[10px] font-mono font-bold text-blue-400 uppercase tracking-wider bg-blue-400/10 px-2 py-0.5 rounded border border-blue-400/20">
                  {activeTechnique.category.toUpperCase()}
                </span>
                <h2 className="text-lg font-semibold text-[#f4f4f5] mt-2">
                  {activeTechnique.name}
                </h2>
              </div>
              <div className="flex items-center gap-2">
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-emerald-500/10 text-emerald-400 border border-emerald-500/20">
                  {activeTechnique.memorySavings}
                </span>
                <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-blue-500/10 text-blue-400 border border-blue-500/20">
                  {activeTechnique.speedMultiplier}
                </span>
              </div>
            </div>

            <p className="text-xs text-zinc-400 leading-relaxed mb-4">
              {activeTechnique.description}
            </p>

            <div className="text-[11px] text-zinc-400 font-mono bg-black/30 p-2.5 rounded border border-zinc-800 mb-4">
              Paper: {activeTechnique.paperReference}
            </div>

            {/* Code preview snippet */}
            <div>
              <div className="text-xs font-medium text-zinc-400 mb-1.5 flex items-center gap-1.5">
                <Code className="w-3.5 h-3.5 text-blue-400" /> Generated PyTorch / Unsloth Implementation
              </div>
              <pre className="bg-zinc-950 p-3.5 rounded text-xs font-mono text-zinc-300 overflow-x-auto border border-zinc-800 leading-relaxed">
                {activeTechnique.codeSnippet}
              </pre>
            </div>
          </div>

          {/* Core Hyperparameter Tuner Form */}
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
            <div className="flex items-center justify-between">
              <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2">
                <Sliders className="w-4 h-4 text-blue-400" /> Fine-Tuning Hyperparameters
              </h3>
              <span className="text-[10px] font-mono text-zinc-500 uppercase">Preset: RTX 4080 Super (16GB)</span>
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs">
              {/* LoRA Rank */}
              <div className="space-y-1.5 bg-black/30 p-3 rounded border border-zinc-800">
                <div className="flex justify-between font-mono text-xs">
                  <span className="text-zinc-400">LoRA Rank (r)</span>
                  <span className="font-bold text-blue-400">{hyperparameters.lora_r}</span>
                </div>
                <input
                  type="range"
                  min="8"
                  max="128"
                  step="8"
                  value={hyperparameters.lora_r}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({
                      ...prev,
                      lora_r: parseInt(e.target.value),
                      lora_alpha: parseInt(e.target.value),
                    }))
                  }
                  className="w-full accent-blue-600 cursor-pointer h-1 bg-zinc-800 rounded"
                />
                <p className="text-[10px] text-zinc-500">Standard ranks: 16 or 32 for general tasks, 64 for complex coding.</p>
              </div>

              {/* LoRA Alpha */}
              <div className="space-y-1.5 bg-black/30 p-3 rounded border border-zinc-800">
                <div className="flex justify-between font-mono text-xs">
                  <span className="text-zinc-400">LoRA Alpha (α)</span>
                  <span className="font-bold text-blue-400">{hyperparameters.lora_alpha}</span>
                </div>
                <input
                  type="range"
                  min="8"
                  max="256"
                  step="8"
                  value={hyperparameters.lora_alpha}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({ ...prev, lora_alpha: parseInt(e.target.value) }))
                  }
                  className="w-full accent-blue-600 cursor-pointer h-1 bg-zinc-800 rounded"
                />
                <p className="text-[10px] text-zinc-500">Scaling constant. Alpha = 16 or 32 provides optimal gradient flow.</p>
              </div>

              {/* Micro Batch Size */}
              <div className="space-y-1.5 bg-black/30 p-3 rounded border border-zinc-800">
                <label className="text-[10px] font-mono text-zinc-500 uppercase block">Micro Batch Size (Per Device)</label>
                <select
                  value={hyperparameters.batch_size}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({ ...prev, batch_size: parseInt(e.target.value) }))
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 font-mono outline-none focus:ring-1 focus:ring-blue-600"
                >
                  <option value={1}>1 (Recommended for 14B models & 16k context)</option>
                  <option value={2}>2 (Recommended for 8B models on RTX 4080)</option>
                  <option value={4}>4 (High speed for 8k context)</option>
                </select>
              </div>

              {/* Gradient Accumulation */}
              <div className="space-y-1.5 bg-black/30 p-3 rounded border border-zinc-800">
                <label className="text-[10px] font-mono text-zinc-500 uppercase block">Gradient Accumulation Steps</label>
                <select
                  value={hyperparameters.gradient_accumulation_steps}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({
                      ...prev,
                      gradient_accumulation_steps: parseInt(e.target.value),
                    }))
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 font-mono outline-none focus:ring-1 focus:ring-blue-600"
                >
                  <option value={2}>2 (Effective batch: 4)</option>
                  <option value={4}>4 (Effective batch: 8)</option>
                  <option value={8}>8 (Effective batch: 16 - High stability)</option>
                  <option value={16}>16 (Effective batch: 32)</option>
                </select>
              </div>

              {/* Learning Rate */}
              <div className="space-y-1.5 bg-black/30 p-3 rounded border border-zinc-800">
                <label className="text-[10px] font-mono text-zinc-500 uppercase block">Learning Rate</label>
                <input
                  type="number"
                  step="0.00001"
                  value={hyperparameters.learning_rate}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({
                      ...prev,
                      learning_rate: parseFloat(e.target.value) || 0.0002,
                    }))
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 font-mono outline-none focus:ring-1 focus:ring-blue-600"
                />
              </div>

              {/* Max Sequence Length */}
              <div className="space-y-1.5 bg-black/30 p-3 rounded border border-zinc-800">
                <label className="text-[10px] font-mono text-zinc-500 uppercase block">Max Sequence Length (Context)</label>
                <select
                  value={hyperparameters.max_seq_length}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({
                      ...prev,
                      max_seq_length: parseInt(e.target.value),
                    }))
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 font-mono outline-none focus:ring-1 focus:ring-blue-600"
                >
                  <option value={2048}>2,048 tokens (Ultra Fast)</option>
                  <option value={4096}>4,096 tokens (Standard)</option>
                  <option value={8192}>8,192 tokens (Extended Instructions)</option>
                  <option value={16384}>16,384 tokens (Full Code & Multi-turn MCP)</option>
                  <option value={32768}>32,768 tokens (Long Document / YaRN)</option>
                </select>
              </div>
            </div>

            {/* Toggle Flags */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-3 pt-2">
              <label className="flex items-center gap-2 p-3 bg-black/30 rounded border border-zinc-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hyperparameters.use_dora}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({ ...prev, use_dora: e.target.checked }))
                  }
                  className="accent-blue-600 rounded"
                />
                <div>
                  <div className="font-medium text-[#f4f4f5] text-xs">Enable DoRA</div>
                  <div className="text-[10px] text-zinc-500">Magnitude / Direction split</div>
                </div>
              </label>

              <label className="flex items-center gap-2 p-3 bg-black/30 rounded border border-zinc-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hyperparameters.use_unsloth_fast_backprop}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({
                      ...prev,
                      use_unsloth_fast_backprop: e.target.checked,
                    }))
                  }
                  className="accent-blue-600 rounded"
                />
                <div>
                  <div className="font-medium text-[#f4f4f5] text-xs">Unsloth Fast Backprop</div>
                  <div className="text-[10px] text-zinc-500">Triton kernel acceleration</div>
                </div>
              </label>

              <label className="flex items-center gap-2 p-3 bg-black/30 rounded border border-zinc-800 cursor-pointer">
                <input
                  type="checkbox"
                  checked={hyperparameters.use_gradient_checkpointing}
                  onChange={(e) =>
                    setHyperparameters((prev) => ({
                      ...prev,
                      use_gradient_checkpointing: e.target.checked,
                    }))
                  }
                  className="accent-blue-600 rounded"
                />
                <div>
                  <div className="font-medium text-[#f4f4f5] text-xs">Gradient Checkpointing</div>
                  <div className="text-[10px] text-zinc-500">Zero OOM for &gt;8k context</div>
                </div>
              </label>
            </div>
          </div>

          {/* Action to proceed */}
          <div className="flex justify-end">
            <button
              onClick={onProceed}
              className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
            >
              <span>Proceed to Dataset & Synthetic Data</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
