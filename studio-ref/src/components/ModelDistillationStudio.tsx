import React, { useState } from "react";
import {
  Sparkles,
  ArrowRight,
  BrainCircuit,
  RefreshCw,
} from "lucide-react";
import { BaseModelInfo, DistillationConfig, TrainingDataSample } from "../types";

interface ModelDistillationStudioProps {
  selectedModel: BaseModelInfo;
  distillationConfig: DistillationConfig;
  setDistillationConfig: React.Dispatch<React.SetStateAction<DistillationConfig>>;
  dataset: TrainingDataSample[];
  setDataset: React.Dispatch<React.SetStateAction<TrainingDataSample[]>>;
  onProceed: () => void;
}

export const ModelDistillationStudio: React.FC<ModelDistillationStudioProps> = ({
  selectedModel,
  distillationConfig,
  setDistillationConfig,
  dataset,
  setDataset,
  onProceed,
}) => {
  const [testPrompt, setTestPrompt] = useState("Explain how to safely deploy an async background task in TypeScript with proper backpressure.");
  const [distillingSample, setDistillingSample] = useState(false);
  const [distilledResult, setDistilledResult] = useState<string | null>(null);

  const handleTestTeacherDistill = async () => {
    setDistillingSample(true);
    try {
      const res = await fetch("/api/distillation/distill-sample", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          teacherPrompt: testPrompt,
          studentArchitecture: selectedModel.name,
          includeReasoning: distillationConfig.includeThoughtChain,
        }),
      });
      const data = await res.json();
      if (data.success) {
        setDistilledResult(data.teacherResponse);
        const newSample: TrainingDataSample = {
          id: `distill-${Date.now()}`,
          instruction: testPrompt,
          output: data.teacherResponse,
          category: "Teacher Distillation",
          difficulty: "Hard",
        };
        setDataset((prev) => [newSample, ...prev]);
      }
    } catch (e: any) {
      console.error(e);
    } finally {
      setDistillingSample(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-blue-400 bg-blue-400/10 border border-blue-400/20 mb-2">
            <BrainCircuit className="w-3.5 h-3.5 text-blue-400" /> MODEL-TO-MODEL DISTILLATION
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Fine-Tune Your Model With Another Model Of Yours
          </h2>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
            Distill knowledge, reasoning chains (<code className="text-blue-300 font-mono">&lt;think&gt;</code>), and expert behaviors from a larger model (e.g. 70B teacher or Gemini) straight into your compact student model ({selectedModel.name}) to run locally on your RTX 4080 Super.
          </p>
        </div>

        <label className="flex items-center gap-3 bg-zinc-950/80 px-3.5 py-2 rounded border border-zinc-800 cursor-pointer">
          <input
            type="checkbox"
            checked={distillationConfig.enabled}
            onChange={(e) =>
              setDistillationConfig((prev) => ({ ...prev, enabled: e.target.checked }))
            }
            className="accent-blue-600 rounded"
          />
          <div>
            <div className="text-xs font-semibold text-[#f4f4f5]">Enable Distillation Engine</div>
            <div className="text-[10px] font-mono text-blue-400">Teacher → Student Pipeline</div>
          </div>
        </label>
      </div>

      {/* Teacher-Student Architecture Map */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-5">
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4 items-center">
          {/* Teacher Model Card */}
          <div className="bg-black/30 p-4 rounded border border-blue-500/30 text-xs space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] font-bold text-blue-400 uppercase tracking-wider">Teacher Model</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-blue-500/10 text-blue-300">Knowledge Source</span>
            </div>
            <select
              value={distillationConfig.teacherModel}
              onChange={(e) =>
                setDistillationConfig((prev) => ({ ...prev, teacherModel: e.target.value }))
              }
              className="w-full bg-zinc-950 border border-zinc-800 rounded p-2 text-zinc-200 font-mono text-xs focus:ring-1 focus:ring-blue-600 outline-none"
            >
              <option value="gemini-3.7-flash">Google Gemini 3.7 Flash (High Reasoning)</option>
              <option value="llama-3.3-70b">Llama 3.3 70B Instruct</option>
              <option value="deepseek-r1-671b">DeepSeek R1 (Full 671B CoT)</option>
              <option value="custom-ollama">Local Custom Ollama Teacher (e.g. my-finetuned-v1)</option>
            </select>
            <p className="text-[11px] text-zinc-400">
              Generates ground truth outputs, synthetic reasoning chains, and self-correction verification.
            </p>
          </div>

          {/* Distillation Transfer Arrow */}
          <div className="text-center space-y-1">
            <div className="text-[10px] font-mono font-bold text-blue-400 uppercase tracking-wider">
              Knowledge Transfer
            </div>
            <div className="h-0.5 bg-gradient-to-r from-blue-500 via-cyan-400 to-emerald-400 w-full rounded my-2" />
            <div className="text-[10px] font-mono text-zinc-400">
              {distillationConfig.includeThoughtChain ? "Chain-of-Thought + Response" : "Direct Response Matching"}
            </div>
          </div>

          {/* Student Model Card */}
          <div className="bg-black/30 p-4 rounded border border-emerald-500/30 text-xs space-y-2">
            <div className="flex items-center justify-between">
              <span className="font-mono text-[10px] font-bold text-emerald-400 uppercase tracking-wider">Student Model (Target)</span>
              <span className="px-1.5 py-0.5 rounded text-[10px] font-mono bg-emerald-500/10 text-emerald-300">Local 4080 Super</span>
            </div>
            <div className="font-semibold text-[#f4f4f5] text-sm">{selectedModel.name}</div>
            <p className="text-[11px] text-zinc-400">
              Learns teacher distribution via Unsloth LoRA/DoRA adapter while preserving low 4.9GB VRAM footprint.
            </p>
          </div>
        </div>

        {/* Distillation Settings */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs pt-2 border-t border-zinc-800">
          <label className="flex items-center gap-3 p-3 bg-black/30 rounded border border-zinc-800 cursor-pointer">
            <input
              type="checkbox"
              checked={distillationConfig.includeThoughtChain}
              onChange={(e) =>
                setDistillationConfig((prev) => ({ ...prev, includeThoughtChain: e.target.checked }))
              }
              className="accent-blue-600 rounded"
            />
            <div>
              <div className="font-medium text-[#f4f4f5]">Extract Deep Reasoning Chains (&lt;think&gt;)</div>
              <div className="text-[10px] text-zinc-500">Forces student to learn step-by-step thinking like DeepSeek R1</div>
            </div>
          </label>

          <div className="p-3 bg-black/30 rounded border border-zinc-800 space-y-1">
            <div className="flex justify-between text-zinc-300 font-mono">
              <span>Teacher Temperature</span>
              <span className="text-blue-400 font-bold">{distillationConfig.temperature}</span>
            </div>
            <input
              type="range"
              min="0.1"
              max="1.0"
              step="0.1"
              value={distillationConfig.temperature}
              onChange={(e) =>
                setDistillationConfig((prev) => ({
                  ...prev,
                  temperature: parseFloat(e.target.value),
                }))
              }
              className="w-full accent-blue-600 h-1 bg-zinc-800 rounded"
            />
          </div>
        </div>

        {/* Live Distillation Playground */}
        <div className="bg-zinc-950 p-4 rounded border border-zinc-800 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#f4f4f5] flex items-center gap-1.5">
              <Sparkles className="w-3.5 h-3.5 text-blue-400" /> Interactive Teacher Probe & Distill
            </span>
            <button
              onClick={handleTestTeacherDistill}
              disabled={distillingSample}
              className="flex items-center gap-1.5 px-3.5 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white cursor-pointer shadow-sm disabled:opacity-50"
            >
              {distillingSample ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Distilling from Teacher...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Probe Teacher Response</span>
                </>
              )}
            </button>
          </div>

          <input
            type="text"
            value={testPrompt}
            onChange={(e) => setTestPrompt(e.target.value)}
            className="w-full bg-[#121214] border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:ring-1 focus:ring-blue-600 outline-none font-mono"
          />

          {distilledResult && (
            <div className="p-3 bg-[#121214] rounded border border-zinc-800 space-y-1">
              <div className="text-[10px] font-mono font-bold text-blue-400 uppercase">
                Distilled Output (Added to Training Dataset):
              </div>
              <pre className="text-[11px] font-mono text-zinc-200 whitespace-pre-wrap leading-relaxed max-h-48 overflow-y-auto">
                {distilledResult}
              </pre>
            </div>
          )}
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onProceed}
          className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
        >
          <span>Proceed to Model Slimming & Fat Shaving</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
