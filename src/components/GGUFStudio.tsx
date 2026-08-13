import React from "react";
import {
  Binary,
  ArrowRight,
  Check,
  Sliders,
} from "lucide-react";
import { BaseModelInfo, GGUFConfig, GGUFQuantType } from "../types";
import { getGGUFSizeEstimate } from "../utils/hardwareCalculator";

interface GGUFStudioProps {
  selectedModel: BaseModelInfo;
  ggufConfig: GGUFConfig;
  setGgufConfig: React.Dispatch<React.SetStateAction<GGUFConfig>>;
  onProceed: () => void;
}

export const GGUFStudio: React.FC<GGUFStudioProps> = ({
  selectedModel,
  ggufConfig,
  setGgufConfig,
  onProceed,
}) => {
  const quantOptions: { type: GGUFQuantType; label: string; desc: string; lossRating: string }[] = [
    {
      type: "Q4_K_M",
      label: "Q4_K_M (Gold Standard)",
      desc: "Medium 4-bit k-quant. Optimal sweet spot between quality, speed, and 16GB VRAM fit.",
      lossRating: "<0.5% Perplexity Loss",
    },
    {
      type: "IQ4_XS",
      label: "IQ4_XS (Importance Matrix 4-bit)",
      desc: "Uses importance matrix quantization for higher fidelity at smaller file size.",
      lossRating: "<0.3% Perplexity Loss",
    },
    {
      type: "Q5_K_M",
      label: "Q5_K_M (High Precision 5-bit)",
      desc: "5-bit medium quant for maximum precision when ample VRAM is available.",
      lossRating: "<0.1% Perplexity Loss",
    },
    {
      type: "Q4_K_S",
      label: "Q4_K_S (Compact 4-bit)",
      desc: "Small 4-bit quant for maximum memory compression.",
      lossRating: "<0.8% Perplexity Loss",
    },
    {
      type: "IQ3_XXS",
      label: "IQ3_XXS (Extreme 3-bit)",
      desc: "Compact 3-bit format to fit 14B models comfortably in 8GB-12GB VRAM.",
      lossRating: "~1.5% Perplexity Loss",
    },
    {
      type: "Q8_0",
      label: "Q8_0 (Near Lossless 8-bit)",
      desc: "8-bit uncompressed precision. Virtually indistinguishable from FP16.",
      lossRating: "0.0% Perplexity Loss",
    },
  ];

  const currentEst = getGGUFSizeEstimate(selectedModel.parametersBillion, ggufConfig.quantization);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 mb-2">
            <Binary className="w-3.5 h-3.5 text-emerald-400" /> GGUF MULTI-TOOL & QUANTIZATION SUITE
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            GGUF Quantization Matrix & RTX 4080 Super Optimization
          </h2>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
            Directly export quantized GGUFs with custom context windows (up to 128k), prompt templates, stop tokens, and full GPU layer offloading (<code className="text-blue-400 font-mono">num_gpu 999</code>) for instant loading in Ollama on Windows.
          </p>
        </div>

        <div className="bg-black/30 px-3.5 py-2 rounded border border-zinc-800 text-xs">
          <div className="text-[10px] font-mono text-zinc-500 uppercase">GGUF File Size</div>
          <div className="text-base font-mono font-bold text-emerald-400">{currentEst.sizeGb} GB</div>
          <div className="text-[10px] text-zinc-500">Fits 16GB GDDR6X ({Math.round((currentEst.ramRequiredGb / 16) * 100)}% VRAM)</div>
        </div>
      </div>

      {/* Quantization Matrix Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {quantOptions.map((q) => {
          const isSelected = ggufConfig.quantization === q.type;
          const est = getGGUFSizeEstimate(selectedModel.parametersBillion, q.type);
          return (
            <div
              key={q.type}
              onClick={() => setGgufConfig((prev) => ({ ...prev, quantization: q.type }))}
              className={`p-4 rounded-lg border transition-all cursor-pointer relative flex flex-col justify-between ${
                isSelected
                  ? "bg-[#18181b] border-blue-500/40 text-blue-400 ring-1 ring-blue-500/20"
                  : "bg-[#121214] border-[#27272a] hover:border-zinc-700 hover:bg-[#18181b]/40 text-zinc-300"
              }`}
            >
              {isSelected && (
                <div className="absolute top-3 right-3 w-4 h-4 rounded-full bg-blue-600 text-white flex items-center justify-center font-bold">
                  <Check className="w-3 h-3 stroke-[3]" />
                </div>
              )}

              <div>
                <div className="font-semibold text-[#f4f4f5] text-xs mb-1 font-mono">{q.label}</div>
                <div className="text-[11px] text-zinc-400 mb-3 leading-relaxed">{q.desc}</div>
              </div>

              <div className="pt-2 border-t border-zinc-800/80 flex items-center justify-between text-xs">
                <span className="font-mono text-emerald-400 font-bold">{est.sizeGb} GB</span>
                <span className="text-[10px] font-mono text-zinc-500">{q.lossRating}</span>
              </div>
            </div>
          );
        })}
      </div>

      {/* GGUF Metadata & Modelfile Parameters Form */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2">
          <Sliders className="w-4 h-4 text-blue-400" /> GGUF Inference Parameters & Modelfile Configuration
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
          <div className="space-y-1.5 bg-black/30 p-3.5 rounded border border-zinc-800">
            <label className="text-[10px] font-mono text-zinc-400 uppercase block">Context Window (num_ctx)</label>
            <select
              value={ggufConfig.contextLength}
              onChange={(e) =>
                setGgufConfig((prev) => ({ ...prev, contextLength: parseInt(e.target.value) }))
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 text-xs font-mono focus:ring-1 focus:ring-blue-600 outline-none"
            >
              <option value={8192}>8,192 tokens</option>
              <option value={16384}>16,384 tokens (Recommended for Coding & MCP)</option>
              <option value={32768}>32,768 tokens (Long context)</option>
              <option value={65536}>65,536 tokens</option>
              <option value={131072}>131,072 tokens (Full Llama 3.1 128k)</option>
            </select>
          </div>

          <div className="space-y-1.5 bg-black/30 p-3.5 rounded border border-zinc-800">
            <label className="text-[10px] font-mono text-zinc-400 uppercase block">Temperature (Sampling)</label>
            <input
              type="number"
              step="0.05"
              value={ggufConfig.temperature}
              onChange={(e) =>
                setGgufConfig((prev) => ({
                  ...prev,
                  temperature: parseFloat(e.target.value) || 0.6,
                }))
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 font-mono text-xs focus:ring-1 focus:ring-blue-600 outline-none"
            />
          </div>

          <div className="space-y-1.5 bg-black/30 p-3.5 rounded border border-zinc-800">
            <label className="text-[10px] font-mono text-zinc-400 uppercase block">GPU Layers Offload (num_gpu)</label>
            <input
              type="number"
              value={ggufConfig.num_gpu_layers}
              onChange={(e) =>
                setGgufConfig((prev) => ({
                  ...prev,
                  num_gpu_layers: parseInt(e.target.value) || 999,
                }))
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 font-mono text-xs focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <span className="text-[10px] text-emerald-400 font-medium block mt-1">
              999 = Full offload to RTX 4080 Super VRAM
            </span>
          </div>
        </div>

        {/* System Prompt */}
        <div className="space-y-1.5">
          <label className="text-[10px] font-mono text-zinc-400 uppercase block">
            Embedded System Instruction for Modelfile
          </label>
          <textarea
            rows={3}
            value={ggufConfig.systemPrompt}
            onChange={(e) =>
              setGgufConfig((prev) => ({ ...prev, systemPrompt: e.target.value }))
            }
            className="w-full bg-zinc-950 border border-zinc-800 rounded p-3 text-xs text-zinc-200 font-mono placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
          />
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onProceed}
          className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
        >
          <span>Launch Live Training Simulator</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
