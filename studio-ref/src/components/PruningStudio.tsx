import React, { useState } from "react";
import {
  Scissors,
  Layers,
  ArrowRight,
  Sparkles,
} from "lucide-react";
import { BaseModelInfo, PruningConfig } from "../types";

interface PruningStudioProps {
  selectedModel: BaseModelInfo;
  pruningConfig: PruningConfig;
  setPruningConfig: React.Dispatch<React.SetStateAction<PruningConfig>>;
  onProceed: () => void;
}

export const PruningStudio: React.FC<PruningStudioProps> = ({
  selectedModel,
  pruningConfig,
  setPruningConfig,
  onProceed,
}) => {
  const [activePruningMethod, setActivePruningMethod] = useState<string>("structured_layer");

  const totalLayers = selectedModel.layers;
  const prunedLayerCount = Math.max(0, pruningConfig.layerPruningRange[1] - pruningConfig.layerPruningRange[0] + 1);
  const remainingLayers = pruningConfig.enabled ? totalLayers - prunedLayerCount : totalLayers;

  const originalSizeGb = selectedModel.baseSizeGb;
  const prunedSizeGb = pruningConfig.enabled
    ? Number((originalSizeGb * (remainingLayers / totalLayers) * 0.95).toFixed(1))
    : originalSizeGb;
  const savedGb = Number((originalSizeGb - prunedSizeGb).toFixed(1));

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-rose-400 bg-rose-500/10 border border-rose-500/20 mb-2">
            <Scissors className="w-3.5 h-3.5 text-rose-400" /> MODEL SLIMMING & FAT SHAVING
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Shave Off the Fat: Structured Pruning & Vocabulary Trimming
          </h2>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
            Eliminate redundant middle layers (ShortGPT angular similarity), prune inactive attention heads, and trim the 128k token vocabulary down to 32k. Reduces VRAM usage and speeds up token generation by 30-40% on RTX 4080 Super.
          </p>
        </div>

        <label className="flex items-center gap-3 bg-zinc-950/80 px-3.5 py-2 rounded border border-zinc-800 cursor-pointer">
          <input
            type="checkbox"
            checked={pruningConfig.enabled}
            onChange={(e) =>
              setPruningConfig((prev) => ({ ...prev, enabled: e.target.checked }))
            }
            className="accent-blue-600 rounded"
          />
          <div>
            <div className="text-xs font-semibold text-[#f4f4f5]">Enable Fat Shaving</div>
            <div className="text-[10px] font-mono text-rose-400">Active Layer/Head Pruning</div>
          </div>
        </label>
      </div>

      {/* Savings Metric Cards */}
      <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
        <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4">
          <div className="text-[10px] font-mono text-zinc-500 uppercase">Total Layers</div>
          <div className="text-xl font-bold font-mono text-[#f4f4f5] mt-1">
            {remainingLayers} <span className="text-xs text-zinc-500 font-normal">/ {totalLayers}</span>
          </div>
          <div className="text-[10px] font-mono text-rose-400 mt-1">
            {pruningConfig.enabled ? `-${prunedLayerCount} redundant layers excised` : "Full 100% layers"}
          </div>
        </div>

        <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4">
          <div className="text-[10px] font-mono text-zinc-500 uppercase">FP16 Weight Size</div>
          <div className="text-xl font-bold font-mono text-emerald-400 mt-1">{prunedSizeGb} GB</div>
          <div className="text-[10px] font-mono text-zinc-500 mt-1">
            {pruningConfig.enabled ? `Down from ${originalSizeGb} GB` : "Standard baseline"}
          </div>
        </div>

        <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4">
          <div className="text-[10px] font-mono text-zinc-500 uppercase">VRAM Shaved</div>
          <div className="text-xl font-bold font-mono text-blue-400 mt-1">
            {pruningConfig.enabled ? `~${savedGb} GB` : "0 GB"}
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">Memory freed for longer context</div>
        </div>

        <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4">
          <div className="text-[10px] font-mono text-zinc-500 uppercase">Throughput Boost</div>
          <div className="text-xl font-bold font-mono text-cyan-400 mt-1">
            {pruningConfig.enabled ? "+35% tok/s" : "1.0x baseline"}
          </div>
          <div className="text-[10px] text-zinc-500 mt-1">Faster inference in Ollama</div>
        </div>
      </div>

      {/* Interactive Layer Topology Map */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
        <div className="flex items-center justify-between">
          <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2">
            <Layers className="w-4 h-4 text-blue-400" /> Transformer Layer Redundancy Map ({selectedModel.name})
          </h3>
          <span className="text-[10px] text-zinc-400 font-mono">
            Red blocks = Redundant layers targeted for pruning
          </span>
        </div>

        {/* Visual Layer Matrix */}
        <div className="grid grid-cols-8 sm:grid-cols-16 gap-1.5 p-4 bg-black/30 rounded border border-zinc-800">
          {Array.from({ length: totalLayers }).map((_, idx) => {
            const isPruned =
              pruningConfig.enabled &&
              idx >= pruningConfig.layerPruningRange[0] &&
              idx <= pruningConfig.layerPruningRange[1];

            return (
              <div
                key={idx}
                title={`Layer ${idx}: ${isPruned ? "Pruned (Excised)" : "Active Transformer Block"}`}
                className={`h-9 rounded flex flex-col items-center justify-center text-[10px] font-mono transition-all ${
                  isPruned
                    ? "bg-rose-950/60 border border-rose-500/60 text-rose-400 opacity-60 scale-95"
                    : "bg-zinc-900 border border-zinc-800 text-zinc-300 hover:border-blue-500"
                }`}
              >
                <span>L{idx}</span>
              </div>
            );
          })}
        </div>

        {/* Pruning Controls */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs pt-2">
          <div className="space-y-1.5 bg-black/30 p-3.5 rounded border border-zinc-800">
            <label className="text-[10px] font-mono text-zinc-400 uppercase block">
              Pruning Start Layer (Middle blocks have highest cosine similarity)
            </label>
            <input
              type="number"
              min="2"
              max={totalLayers - 4}
              value={pruningConfig.layerPruningRange[0]}
              onChange={(e) =>
                setPruningConfig((prev) => ({
                  ...prev,
                  layerPruningRange: [parseInt(e.target.value) || 16, prev.layerPruningRange[1]],
                }))
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 font-mono text-xs focus:ring-1 focus:ring-blue-600 outline-none"
            />
          </div>

          <div className="space-y-1.5 bg-black/30 p-3.5 rounded border border-zinc-800">
            <label className="text-[10px] font-mono text-zinc-400 uppercase block">
              Pruning End Layer
            </label>
            <input
              type="number"
              min="4"
              max={totalLayers - 2}
              value={pruningConfig.layerPruningRange[1]}
              onChange={(e) =>
                setPruningConfig((prev) => ({
                  ...prev,
                  layerPruningRange: [prev.layerPruningRange[0], parseInt(e.target.value) || 23],
                }))
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 font-mono text-xs focus:ring-1 focus:ring-blue-600 outline-none"
            />
          </div>
        </div>

        {/* Healing LoRA Info */}
        <div className="bg-zinc-950 p-3.5 rounded border border-zinc-800 flex items-start gap-3 text-xs">
          <Sparkles className="w-4 h-4 text-blue-400 shrink-0 mt-0.5" />
          <div className="text-zinc-300">
            <span className="font-semibold text-[#f4f4f5]">Automatic Repair LoRA Healing:</span> When layers are excised, Ollama Unsloth Studio automatically runs a 100-step lightweight LoRA healing phase to restore perplexity and bridge the layer gap seamlessly.
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onProceed}
          className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
        >
          <span>Proceed to MoE & Model Merging</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
