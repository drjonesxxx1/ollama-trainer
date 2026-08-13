import React, { useState } from "react";
import {
  UploadCloud,
  Download,
  Copy,
  Check,
  Play,
  Terminal,
  RefreshCw,
} from "lucide-react";
import {
  BaseModelInfo,
  GGUFConfig,
  PruningConfig,
  TrainingHyperparameters,
} from "../types";
import {
  generateModelfile,
  generateUnslothPythonScript,
  generateWindowsPowerShellScript,
} from "../utils/codeGenerators";

interface OllamaDeployerProps {
  selectedModel: BaseModelInfo;
  hyperparameters: TrainingHyperparameters;
  ggufConfig: GGUFConfig;
  pruningConfig: PruningConfig;
  ollamaConnected: boolean;
  checkOllamaConnection: () => void;
  onOpenArena: () => void;
}

export const OllamaDeployer: React.FC<OllamaDeployerProps> = ({
  selectedModel,
  hyperparameters,
  ggufConfig,
  pruningConfig,
  ollamaConnected,
  checkOllamaConnection,
  onOpenArena,
}) => {
  const [modelTag, setModelTag] = useState("my-custom-unsloth-model");
  const [copiedFile, setCopiedFile] = useState<string | null>(null);
  const [activeCodeTab, setActiveCodeTab] = useState<"modelfile" | "python" | "powershell">("modelfile");
  const [isPushingToOllama, setIsPushingToOllama] = useState(false);
  const [pushStatusMessage, setPushStatusMessage] = useState<string | null>(null);

  const modelfileContent = generateModelfile(selectedModel, ggufConfig, modelTag);
  const pythonScript = generateUnslothPythonScript(
    selectedModel,
    hyperparameters,
    ggufConfig,
    pruningConfig,
    "./dataset.json",
    modelTag
  );
  const powerShellScript = generateWindowsPowerShellScript(modelTag);

  const handleCopy = (text: string, fileKey: string) => {
    navigator.clipboard.writeText(text);
    setCopiedFile(fileKey);
    setTimeout(() => setCopiedFile(null), 2000);
  };

  const handleDownload = (filename: string, content: string) => {
    const blob = new Blob([content], { type: "text/plain;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = filename;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handlePushToOllama = async () => {
    setIsPushingToOllama(true);
    setPushStatusMessage("Connecting to local Ollama service (http://localhost:11434)...");

    try {
      const res = await fetch("/api/ollama/proxy", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          endpoint: "http://localhost:11434",
          path: "/api/create",
          method: "POST",
          body: {
            name: modelTag,
            modelfile: modelfileContent,
            stream: false,
          },
        }),
      });

      const data = await res.json();
      if (data.success) {
        setPushStatusMessage(`SUCCESS: Model '${modelTag}' registered in Ollama on your Windows machine!`);
      } else {
        setPushStatusMessage(`Notice: ${data.error || "Ready to execute via local terminal commands below."}`);
      }
    } catch (e: any) {
      setPushStatusMessage("Notice: Use the 1-Click PowerShell script or CLI command below on your Windows machine.");
    } finally {
      setIsPushingToOllama(false);
    }
  };

  return (
    <div className="space-y-6">
      {/* Top Banner */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-blue-400 bg-blue-400/10 border border-blue-400/20 mb-2">
            <UploadCloud className="w-3.5 h-3.5 text-blue-400" /> 1-CLICK WINDOWS 4080 SUPER & OLLAMA EXPORTER
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Export Modelfile & Deploy to Local Ollama
          </h2>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
            Get instant Windows PowerShell automation scripts, standalone Unsloth Python files, and configured Modelfiles with <code className="text-blue-400 font-mono">num_gpu 999</code> for full GPU offloading to your RTX 4080 Super.
          </p>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handlePushToOllama}
            disabled={isPushingToOllama}
            className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer disabled:opacity-50"
          >
            {isPushingToOllama ? (
              <>
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Registering Model in Ollama...</span>
              </>
            ) : (
              <>
                <UploadCloud className="w-4 h-4" />
                <span>One-Click Push to Ollama</span>
              </>
            )}
          </button>
        </div>
      </div>

      {/* Push Status Toast */}
      {pushStatusMessage && (
        <div className="p-3 rounded-lg bg-zinc-950 border border-zinc-800 text-xs font-mono text-blue-400 flex items-center justify-between">
          <span>{pushStatusMessage}</span>
          <button
            onClick={() => setPushStatusMessage(null)}
            className="text-zinc-500 hover:text-zinc-300 text-xs ml-2 cursor-pointer"
          >
            ✕
          </button>
        </div>
      )}

      {/* Model Tag Identifier Input */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="space-y-1">
          <label className="text-[10px] font-mono font-bold text-zinc-400 uppercase tracking-wider">
            Ollama Model Tag Name
          </label>
          <div className="text-xs text-zinc-400">
            This is the tag you will run in your terminal (e.g. <code className="text-blue-400 font-mono">ollama run {modelTag}</code>)
          </div>
        </div>

        <input
          type="text"
          value={modelTag}
          onChange={(e) => setModelTag(e.target.value.toLowerCase().replace(/[^a-z0-9-_:]/g, "-"))}
          className="bg-zinc-950 border border-zinc-800 rounded px-3 py-1.5 text-xs font-mono text-blue-400 focus:ring-1 focus:ring-blue-600 outline-none w-full sm:w-80"
        />
      </div>

      {/* Code Export Tabs */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
        <div className="flex flex-wrap items-center justify-between gap-3 border-b border-[#27272a] pb-3">
          <div className="flex items-center space-x-1.5 text-xs">
            <button
              onClick={() => setActiveCodeTab("modelfile")}
              className={`px-3 py-1.5 rounded font-medium transition-all cursor-pointer ${
                activeCodeTab === "modelfile"
                  ? "bg-zinc-800 text-blue-400 font-semibold border border-blue-500/30 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              Modelfile
            </button>
            <button
              onClick={() => setActiveCodeTab("python")}
              className={`px-3 py-1.5 rounded font-medium transition-all cursor-pointer ${
                activeCodeTab === "python"
                  ? "bg-zinc-800 text-blue-400 font-semibold border border-blue-500/30 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              train_unsloth.py
            </button>
            <button
              onClick={() => setActiveCodeTab("powershell")}
              className={`px-3 py-1.5 rounded font-medium transition-all cursor-pointer ${
                activeCodeTab === "powershell"
                  ? "bg-zinc-800 text-blue-400 font-semibold border border-blue-500/30 shadow-sm"
                  : "text-zinc-400 hover:text-zinc-200"
              }`}
            >
              train_and_quantize.ps1 (Windows 4080)
            </button>
          </div>

          <div className="flex items-center gap-2">
            <button
              onClick={() => {
                const text =
                  activeCodeTab === "modelfile"
                    ? modelfileContent
                    : activeCodeTab === "python"
                    ? pythonScript
                    : powerShellScript;
                handleCopy(text, activeCodeTab);
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 cursor-pointer"
            >
              {copiedFile === activeCodeTab ? (
                <>
                  <Check className="w-3.5 h-3.5 text-emerald-400" />
                  <span>Copied!</span>
                </>
              ) : (
                <>
                  <Copy className="w-3.5 h-3.5" />
                  <span>Copy Code</span>
                </>
              )}
            </button>

            <button
              onClick={() => {
                if (activeCodeTab === "modelfile") handleDownload("Modelfile", modelfileContent);
                else if (activeCodeTab === "python") handleDownload("train_unsloth.py", pythonScript);
                else handleDownload("train_and_quantize.ps1", powerShellScript);
              }}
              className="flex items-center gap-1.5 px-3 py-1.5 rounded text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 cursor-pointer"
            >
              <Download className="w-3.5 h-3.5" />
              <span>Download File</span>
            </button>
          </div>
        </div>

        {/* Code Content Display */}
        <pre className="bg-zinc-950 p-4 rounded text-xs font-mono text-zinc-300 overflow-x-auto border border-zinc-800 leading-relaxed max-h-96">
          {activeCodeTab === "modelfile" && modelfileContent}
          {activeCodeTab === "python" && pythonScript}
          {activeCodeTab === "powershell" && powerShellScript}
        </pre>
      </div>

      {/* Windows RTX 4080 Super Terminal Cheat-Sheet */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
        <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2">
          <Terminal className="w-4 h-4 text-emerald-400" /> Windows RTX 4080 Super Terminal Commands
        </h3>

        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-xs font-mono">
          <div className="bg-black/30 p-3.5 rounded border border-zinc-800 space-y-1">
            <div className="text-[10px] text-zinc-500 uppercase font-mono">
              1. Run Training & Export GGUF
            </div>
            <div className="text-emerald-400 font-mono">python train_unsloth.py</div>
            <div className="text-[10px] text-zinc-500 font-sans">
              Takes ~5-12 mins on RTX 4080 Super with 16k context
            </div>
          </div>

          <div className="bg-black/30 p-3.5 rounded border border-zinc-800 space-y-1">
            <div className="text-[10px] text-zinc-500 uppercase font-mono">
              2. Register Modelfile in Ollama
            </div>
            <div className="text-emerald-400 font-mono">ollama create {modelTag} -f Modelfile</div>
            <div className="text-[10px] text-zinc-500 font-sans">
              Instant registration using quantized GGUF
            </div>
          </div>

          <div className="bg-black/30 p-3.5 rounded border border-zinc-800 space-y-1">
            <div className="text-[10px] text-zinc-500 uppercase font-mono">
              3. Run Local Interactive Chat
            </div>
            <div className="text-emerald-400 font-mono">ollama run {modelTag}</div>
            <div className="text-[10px] text-zinc-500 font-sans">
              Executes with full GPU offload (100% VRAM)
            </div>
          </div>

          <div className="bg-black/30 p-3.5 rounded border border-zinc-800 space-y-1">
            <div className="text-[10px] text-zinc-500 uppercase font-mono">
              4. Test MCP Tool Execution
            </div>
            <div className="text-emerald-400 font-mono">
              curl http://localhost:11434/api/generate -d '{`{"model": "${modelTag}", "prompt": "Call filesystem read_file on src/App.tsx"}`}'
            </div>
            <div className="text-[10px] text-zinc-500 font-sans">
              Outputs valid JSON function call
            </div>
          </div>
        </div>
      </div>

      <div className="flex justify-end">
        <button
          onClick={onOpenArena}
          className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
        >
          <span>Open Interactive Model Arena Playground</span>
          <Play className="w-3.5 h-3.5" />
        </button>
      </div>
    </div>
  );
};
