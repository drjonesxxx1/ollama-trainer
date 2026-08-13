import React, { useState } from "react";
import {
  Database,
  Upload,
  Plus,
  Trash2,
  Sparkles,
  RefreshCw,
  ArrowRight,
  Download,
} from "lucide-react";
import { TrainingDataSample } from "../types";

interface DatasetStudioProps {
  dataset: TrainingDataSample[];
  setDataset: React.Dispatch<React.SetStateAction<TrainingDataSample[]>>;
  onProceed: () => void;
}

export const DatasetStudio: React.FC<DatasetStudioProps> = ({
  dataset,
  setDataset,
  onProceed,
}) => {
  const [activeView, setActiveView] = useState<"samples" | "json_editor" | "synthetic_generator">("samples");
  const [jsonText, setJsonText] = useState<string>(() => JSON.stringify(dataset, null, 2));
  const [domainPrompt, setDomainPrompt] = useState<string>("MCP function-calling and Python data engineering tasks");
  const [samplesCount, setSamplesCount] = useState<number>(5);
  const [taskFormat, setTaskFormat] = useState<string>("alpaca");
  const [generating, setGenerating] = useState<boolean>(false);
  const [filterCategory, setFilterCategory] = useState<string>("all");

  // New manual item
  const [newItemInstruction, setNewItemInstruction] = useState("");
  const [newItemInput, setNewItemInput] = useState("");
  const [newItemOutput, setNewItemOutput] = useState("");

  const handleAddNewItem = () => {
    if (!newItemInstruction.trim() || !newItemOutput.trim()) return;
    const newSample: TrainingDataSample = {
      id: `sample-${Date.now()}`,
      instruction: newItemInstruction.trim(),
      input: newItemInput.trim() || undefined,
      output: newItemOutput.trim(),
      category: "Custom",
      difficulty: "Medium",
    };
    setDataset((prev) => [newSample, ...prev]);
    setNewItemInstruction("");
    setNewItemInput("");
    setNewItemOutput("");
  };

  const handleDeleteItem = (id: string) => {
    setDataset((prev) => prev.filter((item) => item.id !== id));
  };

  const handleApplyJsonEditor = () => {
    try {
      const parsed = JSON.parse(jsonText);
      if (Array.isArray(parsed)) {
        setDataset(parsed);
      }
    } catch (e: any) {
      console.error(e);
    }
  };

  const handleGenerateSyntheticData = async () => {
    setGenerating(true);
    try {
      const res = await fetch("/api/dataset/generate", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          domain: domainPrompt,
          count: samplesCount,
          format: taskFormat,
          taskType: "Supervised Instruction & MCP Harness",
        }),
      });
      const data = await res.json();
      if (data.success && Array.isArray(data.data)) {
        setDataset((prev) => [...data.data, ...prev]);
        setJsonText(JSON.stringify([...data.data, ...dataset], null, 2));
      }
    } catch (e: any) {
      console.error(e);
    } finally {
      setGenerating(false);
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const content = event.target?.result as string;
        let parsed: any[] = [];
        if (file.name.endsWith(".jsonl")) {
          parsed = content
            .split("\n")
            .filter((line) => line.trim())
            .map((line) => JSON.parse(line));
        } else if (file.name.endsWith(".json")) {
          const raw = JSON.parse(content);
          parsed = Array.isArray(raw) ? raw : [raw];
        } else if (file.name.endsWith(".csv")) {
          const lines = content.split("\n").filter((l) => l.trim());
          for (let i = 1; i < lines.length; i++) {
            const values = lines[i].split(",");
            parsed.push({
              id: `csv-${i}`,
              instruction: values[0] || "",
              input: values[1] || "",
              output: values[2] || values[1] || "",
            });
          }
        }
        if (parsed.length > 0) {
          setDataset((prev) => [...parsed, ...prev]);
          setJsonText(JSON.stringify([...parsed, ...dataset], null, 2));
        }
      } catch (err: any) {
        console.error(err);
      }
    };
    reader.readAsText(file);
  };

  const filteredDataset = filterCategory === "all"
    ? dataset
    : dataset.filter((d) => d.category === filterCategory || (filterCategory === "mcp" && d.isMcpSample));

  const totalTokensEst = dataset.reduce(
    (acc, item) => acc + (item.instruction.length + (item.input?.length || 0) + item.output.length) / 4,
    0
  );

  return (
    <div className="space-y-6">
      {/* Top Banner & Stats */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-emerald-400 bg-emerald-500/10 border border-emerald-500/20 mb-2">
            <Database className="w-3.5 h-3.5" /> STRUCTURED DATASET STUDIO
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Import, Format & Synthesize Training Data
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Feed structured JSON, JSONL, CSV, or generate custom synthetic pairs with Gemini AI.
          </p>
        </div>

        {/* Dataset Stats Strip */}
        <div className="flex items-center gap-3 text-xs">
          <div className="bg-black/30 px-3.5 py-2 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Total Samples</div>
            <div className="text-sm font-mono font-bold text-blue-400">{dataset.length} pairs</div>
          </div>
          <div className="bg-black/30 px-3.5 py-2 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Est. Tokens</div>
            <div className="text-sm font-mono font-bold text-zinc-200">{Math.round(totalTokensEst).toLocaleString()}</div>
          </div>
          <div className="bg-black/30 px-3.5 py-2 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Format</div>
            <div className="text-sm font-mono font-bold text-emerald-400 uppercase">Alpaca / Tool</div>
          </div>
        </div>
      </div>

      {/* View Switcher Tabs & Actions */}
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div className="flex items-center space-x-1.5 bg-[#121214] p-1 rounded border border-[#27272a] text-xs">
          <button
            onClick={() => setActiveView("samples")}
            className={`px-3 py-1.5 rounded font-medium transition-all cursor-pointer ${
              activeView === "samples"
                ? "bg-zinc-800 text-blue-400 font-semibold border border-blue-500/30 shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Sample Explorer ({dataset.length})
          </button>
          <button
            onClick={() => {
              setJsonText(JSON.stringify(dataset, null, 2));
              setActiveView("json_editor");
            }}
            className={`px-3 py-1.5 rounded font-medium transition-all cursor-pointer ${
              activeView === "json_editor"
                ? "bg-zinc-800 text-blue-400 font-semibold border border-blue-500/30 shadow-sm"
                : "text-zinc-400 hover:text-zinc-200"
            }`}
          >
            Raw JSON / JSONL Editor
          </button>
          <button
            onClick={() => setActiveView("synthetic_generator")}
            className={`flex items-center gap-1.5 px-3 py-1.5 rounded font-medium transition-all cursor-pointer ${
              activeView === "synthetic_generator"
                ? "bg-blue-600 text-white font-semibold shadow-sm"
                : "text-blue-400 hover:bg-blue-950/30"
            }`}
          >
            <Sparkles className="w-3.5 h-3.5" />
            <span>AI Synthetic Generator</span>
          </button>
        </div>

        {/* Upload Button */}
        <div className="flex items-center gap-2">
          <label className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 cursor-pointer">
            <Upload className="w-3.5 h-3.5 text-blue-400" />
            <span>Import JSON / JSONL / CSV</span>
            <input
              type="file"
              accept=".json,.jsonl,.csv"
              onChange={handleFileUpload}
              className="hidden"
            />
          </label>

          <button
            onClick={() => {
              const blob = new Blob([JSON.stringify(dataset, null, 2)], { type: "application/json" });
              const url = URL.createObjectURL(blob);
              const a = document.createElement("a");
              a.href = url;
              a.download = "training_dataset.json";
              a.click();
            }}
            className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 cursor-pointer"
          >
            <Download className="w-3.5 h-3.5 text-zinc-400" />
            <span>Export JSON</span>
          </button>
        </div>
      </div>

      {/* VIEW 1: Samples List */}
      {activeView === "samples" && (
        <div className="space-y-4">
          {/* Quick Manual Add Form */}
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4 space-y-3">
            <div className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-wider flex items-center gap-1.5">
              <Plus className="w-3.5 h-3.5 text-blue-400" /> Add Custom Instruction Pair
            </div>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-3 text-xs">
              <input
                type="text"
                placeholder="Instruction (e.g. Write a Python script to query PostgreSQL MCP...)"
                value={newItemInstruction}
                onChange={(e) => setNewItemInstruction(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
              />
              <input
                type="text"
                placeholder="Optional Input Context / Schema"
                value={newItemInput}
                onChange={(e) => setNewItemInput(e.target.value)}
                className="bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
              />
            </div>
            <textarea
              placeholder="Target Response / Assistant Output (including JSON tool-calls or reasoning)"
              rows={2}
              value={newItemOutput}
              onChange={(e) => setNewItemOutput(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded p-2.5 text-xs text-zinc-200 placeholder-zinc-600 font-mono focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <div className="flex justify-end">
              <button
                onClick={handleAddNewItem}
                className="px-3.5 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white cursor-pointer shadow-sm"
              >
                Insert Sample
              </button>
            </div>
          </div>

          {/* Samples Table / Cards */}
          <div className="space-y-3">
            {filteredDataset.map((sample, idx) => (
              <div
                key={sample.id || idx}
                className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4 space-y-2 text-xs hover:border-zinc-700 transition-colors"
              >
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="font-mono text-zinc-500">#{idx + 1}</span>
                    {sample.category && (
                      <span className="px-2 py-0.5 rounded text-[10px] bg-zinc-900 text-zinc-300 font-mono font-medium border border-zinc-800">
                        {sample.category}
                      </span>
                    )}
                    {sample.isMcpSample && (
                      <span className="px-2 py-0.5 rounded text-[10px] bg-blue-500/10 text-blue-400 font-mono font-medium border border-blue-500/20">
                        MCP Plugin Pair
                      </span>
                    )}
                  </div>
                  <button
                    onClick={() => handleDeleteItem(sample.id)}
                    className="text-zinc-500 hover:text-rose-400 cursor-pointer p-1"
                  >
                    <Trash2 className="w-3.5 h-3.5" />
                  </button>
                </div>

                <div className="font-medium text-zinc-200">
                  <span className="text-blue-400 mr-1.5 font-mono">User:</span> {sample.instruction}
                </div>

                {sample.input && (
                  <div className="text-zinc-400 bg-black/30 p-2 rounded border border-zinc-800 font-mono text-[11px]">
                    <span className="text-zinc-500 block text-[10px] uppercase font-mono">Input Context:</span>
                    {sample.input}
                  </div>
                )}

                <div className="bg-zinc-950 p-3 rounded border border-zinc-800 font-mono text-[11px] text-zinc-300 whitespace-pre-wrap">
                  <span className="text-emerald-400 block text-[10px] uppercase font-mono font-bold mb-1">
                    Assistant Output:
                  </span>
                  {sample.output}
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* VIEW 2: Raw JSON Editor */}
      {activeView === "json_editor" && (
        <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4 space-y-3">
          <div className="flex items-center justify-between">
            <span className="text-xs font-semibold text-[#f4f4f5]">
              Direct JSON Array Representation
            </span>
            <button
              onClick={handleApplyJsonEditor}
              className="px-3.5 py-1.5 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white cursor-pointer shadow-sm"
            >
              Apply JSON Changes
            </button>
          </div>
          <textarea
            rows={18}
            value={jsonText}
            onChange={(e) => setJsonText(e.target.value)}
            className="w-full bg-zinc-950 border border-zinc-800 rounded p-4 font-mono text-xs text-zinc-200 leading-relaxed focus:outline-none focus:ring-1 focus:ring-blue-600"
          />
        </div>
      )}

      {/* VIEW 3: AI Synthetic Generator */}
      {activeView === "synthetic_generator" && (
        <div className="bg-[#18181b]/50 border border-blue-500/30 rounded-xl p-5 space-y-4">
          <div className="max-w-2xl">
            <h3 className="text-base font-semibold text-[#f4f4f5] flex items-center gap-2">
              <Sparkles className="w-4 h-4 text-blue-400" /> Gemini Synthetic Dataset Generator
            </h3>
            <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
              Generate hundreds of diverse, edge-case instruction pairs, tool calls, and structured dialogues directly into your fine-tuning dataset using Google GenAI.
            </p>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-3 gap-4 text-xs">
            <div className="md:col-span-2 space-y-1.5">
              <label className="text-[10px] font-mono text-zinc-500 uppercase block">Domain / Target Task Description</label>
              <input
                type="text"
                value={domainPrompt}
                onChange={(e) => setDomainPrompt(e.target.value)}
                placeholder="e.g. MCP filesystem file editing, SQL schema migration, reasoning chains..."
                className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
              />
            </div>

            <div className="space-y-1.5">
              <label className="text-[10px] font-mono text-zinc-500 uppercase block">Number of Samples to Generate</label>
              <select
                value={samplesCount}
                onChange={(e) => setSamplesCount(parseInt(e.target.value))}
                className="w-full bg-zinc-950 border border-zinc-800 rounded px-3 py-2 text-zinc-200 font-mono outline-none focus:ring-1 focus:ring-blue-600"
              >
                <option value={5}>5 High-Quality Pairs</option>
                <option value={10}>10 Diverse Pairs</option>
                <option value={20}>20 Edge-Case Pairs</option>
              </select>
            </div>
          </div>

          <div className="flex justify-end pt-2">
            <button
              onClick={handleGenerateSyntheticData}
              disabled={generating}
              className="flex items-center gap-2 px-4 py-2 rounded font-medium text-xs bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer disabled:opacity-50"
            >
              {generating ? (
                <>
                  <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                  <span>Synthesizing Training Examples...</span>
                </>
              ) : (
                <>
                  <Sparkles className="w-3.5 h-3.5" />
                  <span>Generate Synthetic Pairs</span>
                </>
              )}
            </button>
          </div>
        </div>
      )}

      {/* Footer Navigation */}
      <div className="flex justify-end pt-4 border-t border-[#27272a]">
        <button
          onClick={onProceed}
          className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
        >
          <span>Proceed to MCP Plugins Harness</span>
          <ArrowRight className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
