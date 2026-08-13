import React, { useState } from "react";
import {
  Network,
  Plus,
  Trash2,
  ArrowRight,
  Code,
} from "lucide-react";
import { BaseModelInfo, MoEConfig } from "../types";
import { generateMergeKitConfig } from "../utils/codeGenerators";

interface MoEStudioProps {
  selectedModel: BaseModelInfo;
  moeConfig: MoEConfig;
  setMoeConfig: React.Dispatch<React.SetStateAction<MoEConfig>>;
  onProceed: () => void;
}

export const MoEStudio: React.FC<MoEStudioProps> = ({
  selectedModel,
  moeConfig,
  setMoeConfig,
  onProceed,
}) => {
  const [newExpertName, setNewExpertName] = useState("");
  const [newExpertModelId, setNewExpertModelId] = useState("");
  const [newExpertSpecialization, setNewExpertSpecialization] = useState("");

  const handleAddExpert = () => {
    if (!newExpertName.trim()) return;
    setMoeConfig((prev) => ({
      ...prev,
      expertSources: [
        ...prev.expertSources,
        {
          name: newExpertName.trim(),
          modelId: newExpertModelId.trim() || selectedModel.huggingFaceId,
          weight: 0.5,
          specialization: newExpertSpecialization.trim() || "General Reasoning & Tools",
        },
      ],
      numExperts: prev.expertSources.length + 1,
    }));
    setNewExpertName("");
    setNewExpertModelId("");
    setNewExpertSpecialization("");
  };

  const handleDeleteExpert = (index: number) => {
    setMoeConfig((prev) => ({
      ...prev,
      expertSources: prev.expertSources.filter((_, i) => i !== index),
      numExperts: Math.max(2, prev.expertSources.length - 1),
    }));
  };

  const mergeKitYaml = generateMergeKitConfig(moeConfig, selectedModel.huggingFaceId);

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-cyan-400 bg-cyan-500/10 border border-cyan-500/20 mb-2">
            <Network className="w-3.5 h-3.5 text-cyan-400" /> MOE (MIXTURE OF EXPERTS) & MERGEKIT
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Add Experts & Merge Multiple Fine-Tuned Checkpoints
          </h2>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
            Upcycle your dense {selectedModel.parametersBillion}B model into an MoE (e.g. 4x8B or 8x8B with top-2 router), or fuse specialized weights (Coding + MCP Tool Calling + Mathematics) using DARE-TIES and SLERP algorithms.
          </p>
        </div>

        <label className="flex items-center gap-3 bg-zinc-950/80 px-3.5 py-2 rounded border border-zinc-800 cursor-pointer">
          <input
            type="checkbox"
            checked={moeConfig.enabled}
            onChange={(e) =>
              setMoeConfig((prev) => ({ ...prev, enabled: e.target.checked }))
            }
            className="accent-blue-600 rounded"
          />
          <div>
            <div className="text-xs font-semibold text-[#f4f4f5]">Enable MoE / Merging</div>
            <div className="text-[10px] font-mono text-cyan-400">Active Multi-Expert Routing</div>
          </div>
        </label>
      </div>

      {/* Main Grid: Architecture Settings + Visual Router */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: MoE / Merge Config */}
        <div className="space-y-4">
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
            <h3 className="text-xs font-mono font-bold text-zinc-400 uppercase tracking-wider">
              Merge & MoE Method
            </h3>

            <div className="space-y-2 text-xs">
              <label className="text-[10px] font-mono text-zinc-400 uppercase block">Algorithm</label>
              <select
                value={moeConfig.method}
                onChange={(e) =>
                  setMoeConfig((prev) => ({ ...prev, method: e.target.value as any }))
                }
                className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-2 text-zinc-200 text-xs font-mono focus:ring-1 focus:ring-blue-600 outline-none"
              >
                <option value="moefication">MoEfication (Dense → Sparse MoE with Router)</option>
                <option value="dare_ties">DARE-TIES (Extreme Delta Rescaling & Sign Fix)</option>
                <option value="slerp">SLERP (Spherical Linear Interpolation)</option>
                <option value="passthrough_franken">Frankenmerging / Passthrough Layer Slicing</option>
                <option value="task_arithmetic">Task Arithmetic (Directional Vector Addition)</option>
              </select>
            </div>

            <div className="grid grid-cols-2 gap-3 text-xs">
              <div className="space-y-1">
                <label className="text-[10px] font-mono text-zinc-400 uppercase block">Top-K Active</label>
                <select
                  value={moeConfig.topK}
                  onChange={(e) =>
                    setMoeConfig((prev) => ({ ...prev, topK: parseInt(e.target.value) }))
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 text-xs font-mono focus:ring-1 focus:ring-blue-600 outline-none"
                >
                  <option value={1}>Top 1 Expert</option>
                  <option value={2}>Top 2 Experts (Standard)</option>
                  <option value={4}>Top 4 Experts</option>
                </select>
              </div>

              <div className="space-y-1">
                <label className="text-[10px] font-mono text-zinc-400 uppercase block">Gating Router</label>
                <select
                  value={moeConfig.routerType}
                  onChange={(e) =>
                    setMoeConfig((prev) => ({ ...prev, routerType: e.target.value as any }))
                  }
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 text-xs font-mono focus:ring-1 focus:ring-blue-600 outline-none"
                >
                  <option value="softmax">Softmax Gating</option>
                  <option value="sinkhorn">Sinkhorn Balanced</option>
                  <option value="switch">Switch Transformer</option>
                </select>
              </div>
            </div>
          </div>

          {/* Add Expert Form */}
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-3 text-xs">
            <h3 className="font-semibold text-[#f4f4f5] flex items-center gap-1.5 text-xs">
              <Plus className="w-3.5 h-3.5 text-blue-400" /> Add Expert Source Model
            </h3>
            <input
              type="text"
              placeholder="Expert Name (e.g. MCP-Tool-Expert)"
              value={newExpertName}
              onChange={(e) => setNewExpertName(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <input
              type="text"
              placeholder="HuggingFace ID or Local Checkpoint path"
              value={newExpertModelId}
              onChange={(e) => setNewExpertModelId(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <input
              type="text"
              placeholder="Specialization (e.g. JSON Tool Calling & MCP)"
              value={newExpertSpecialization}
              onChange={(e) => setNewExpertSpecialization(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <button
              onClick={handleAddExpert}
              className="w-full py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white cursor-pointer shadow-sm"
            >
              Add Expert Block
            </button>
          </div>
        </div>

        {/* Right: Expert Roster & Visual Gating Network */}
        <div className="lg:col-span-2 space-y-6">
          {/* Visual Gating Diagram */}
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
            <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2">
              <Network className="w-4 h-4 text-blue-400" /> MoE Router & Expert Dispatch Topology
            </h3>

            {/* Visual Flow diagram */}
            <div className="bg-black/30 p-5 rounded border border-zinc-800 space-y-4">
              <div className="flex justify-center">
                <div className="bg-zinc-900 border border-zinc-800 px-3.5 py-1.5 rounded text-center text-xs font-mono text-zinc-300">
                  <span>Input Token Stream</span>
                </div>
              </div>

              {/* Router Node */}
              <div className="flex justify-center">
                <div className="bg-blue-950/40 border border-blue-500/40 px-5 py-1.5 rounded text-center text-xs font-mono font-medium text-blue-400 shadow-sm">
                  <span>{moeConfig.routerType.toUpperCase()} Gating Router (Top-{moeConfig.topK})</span>
                </div>
              </div>

              {/* Experts Grid */}
              <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-3 gap-3 pt-2">
                {moeConfig.expertSources.map((exp, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-zinc-950 border border-zinc-800 rounded space-y-1 text-xs relative group"
                  >
                    <div className="flex items-center justify-between">
                      <span className="font-semibold text-[#f4f4f5] font-mono text-[11px]">
                        Expert #{idx + 1}: {exp.name}
                      </span>
                      <button
                        onClick={() => handleDeleteExpert(idx)}
                        className="text-zinc-500 hover:text-rose-400 p-0.5 cursor-pointer opacity-0 group-hover:opacity-100 transition-opacity"
                      >
                        <Trash2 className="w-3.5 h-3.5" />
                      </button>
                    </div>
                    <div className="text-[11px] text-zinc-400 truncate">{exp.specialization}</div>
                    <div className="flex items-center justify-between text-[10px] font-mono text-zinc-500 pt-1 border-t border-zinc-800/80">
                      <span>Weight: {exp.weight}</span>
                      <span className="text-emerald-400">Active</span>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* MergeKit YAML Preview */}
            <div>
              <div className="text-xs font-medium text-zinc-400 mb-1.5 flex items-center gap-1.5">
                <Code className="w-3.5 h-3.5 text-blue-400" /> Generated MergeKit / MoE Config YAML
              </div>
              <pre className="bg-zinc-950 p-3 rounded text-xs font-mono text-zinc-300 overflow-x-auto border border-zinc-800">
                {mergeKitYaml}
              </pre>
            </div>
          </div>

          <div className="flex justify-end">
            <button
              onClick={onProceed}
              className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
            >
              <span>Proceed to GGUF Quantization Matrix</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
