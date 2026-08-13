import React from "react";
import { Check, Zap, ArrowRight, ShieldCheck, AlertTriangle } from "lucide-react";
import { BaseModelInfo } from "../types";
import { BASE_MODELS } from "../data/models";

interface ModelSelectorProps {
  selectedModel: BaseModelInfo;
  onSelectModel: (model: BaseModelInfo) => void;
  onProceed: () => void;
}

export const ModelSelector: React.FC<ModelSelectorProps> = ({
  selectedModel,
  onSelectModel,
  onProceed,
}) => {
  return (
    <div className="space-y-6">
      {/* Intro Banner */}
      <div className="bg-[#18181b]/60 border border-[#27272a] rounded-xl p-6 relative overflow-hidden">
        <div className="max-w-3xl relative z-10">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded text-xs font-mono text-blue-400 bg-blue-400/10 border border-blue-400/20 mb-3">
            <Zap className="w-3.5 h-3.5 text-blue-400" /> RTX 4080 SUPER (16GB VRAM) OPTIMIZED ARCHITECTURES
          </div>
          <h2 className="text-xl font-semibold text-[#f4f4f5] tracking-tight">
            Select Your Foundation Model
          </h2>
          <p className="text-xs text-zinc-400 mt-2 leading-relaxed">
            Choose from state-of-the-art open models natively accelerated with Unsloth Triton kernels,
            4-bit NormalFloat quantization, and FlashAttention-2. All models below support full MCP tool-calling,
            GGUF quantization, and direct export to your local Ollama instance.
          </p>
        </div>
      </div>

      {/* Model Cards Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
        {BASE_MODELS.map((model) => {
          const isSelected = selectedModel.id === model.id;
          return (
            <div
              key={model.id}
              id={`model-card-${model.id}`}
              onClick={() => onSelectModel(model)}
              className={`rounded-xl p-5 border transition-all cursor-pointer relative flex flex-col justify-between ${
                isSelected
                  ? "bg-[#18181b] border-blue-500/50 shadow-lg shadow-blue-500/5 ring-1 ring-blue-500/30"
                  : "bg-[#121214] border-[#27272a] hover:border-zinc-700 hover:bg-[#18181b]/50"
              }`}
            >
              {/* Selected Checkmark */}
              {isSelected && (
                <div className="absolute top-4 right-4 w-5 h-5 rounded bg-blue-600 text-white flex items-center justify-center font-bold shadow-sm">
                  <Check className="w-3.5 h-3.5 stroke-[2.5]" />
                </div>
              )}

              <div>
                {/* Header */}
                <div className="flex items-center gap-2 mb-2">
                  <span className="px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider font-mono bg-zinc-900 text-zinc-300 border border-zinc-800">
                    {model.architecture}
                  </span>
                  {model.recommendedFor4080Super ? (
                    <span className="flex items-center gap-1 text-[10px] font-mono font-medium text-emerald-400">
                      <ShieldCheck className="w-3.5 h-3.5" /> 16GB Ready
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-[10px] font-mono font-medium text-amber-400">
                      <AlertTriangle className="w-3.5 h-3.5" /> High VRAM
                    </span>
                  )}
                </div>

                <h3 className="font-semibold text-base text-[#f4f4f5] mb-1">
                  {model.name}
                </h3>
                <div className="text-xs font-mono text-zinc-500 mb-3 truncate">
                  {model.huggingFaceId}
                </div>

                <p className="text-xs text-zinc-400 line-clamp-3 mb-4 leading-relaxed">
                  {model.description}
                </p>
              </div>

              {/* Specs Badge Strip */}
              <div className="pt-3 border-t border-zinc-800/80 grid grid-cols-3 gap-2 text-center text-xs">
                <div className="bg-black/30 rounded p-2 border border-zinc-800">
                  <div className="text-[10px] font-mono text-zinc-500 uppercase">Params</div>
                  <div className="font-mono font-medium text-zinc-200">{model.parametersBillion}B</div>
                </div>
                <div className="bg-black/30 rounded p-2 border border-zinc-800">
                  <div className="text-[10px] font-mono text-zinc-500 uppercase">Q4 GGUF</div>
                  <div className="font-mono font-medium text-emerald-400">{model.q4SizeGb} GB</div>
                </div>
                <div className="bg-black/30 rounded p-2 border border-zinc-800">
                  <div className="text-[10px] font-mono text-zinc-500 uppercase">Context</div>
                  <div className="font-mono font-medium text-zinc-200">
                    {model.defaultContext > 32768 ? "128k" : `${model.defaultContext / 1024}k`}
                  </div>
                </div>
              </div>
            </div>
          );
        })}
      </div>

      {/* Selected Model Summary Action */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="text-[10px] font-mono text-zinc-500 uppercase tracking-wider">
            Active Base Target
          </div>
          <div className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2 mt-0.5">
            <span>{selectedModel.name}</span>
            <span className="text-xs font-mono font-normal text-zinc-400">
              ({selectedModel.parametersBillion}B parameters • {selectedModel.layers} layers • {selectedModel.vocabSize.toLocaleString()} vocab)
            </span>
          </div>
        </div>

        <button
          id="btn-proceed-to-techniques"
          onClick={onProceed}
          className="flex items-center gap-2 px-4 py-2 rounded font-medium text-xs bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
        >
          <span>Configure Training Techniques</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
