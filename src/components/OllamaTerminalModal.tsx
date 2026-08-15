import React, { useState, useEffect, useRef } from "react";
import {
  Terminal as TerminalIcon,
  X,
  UploadCloud,
  RefreshCw,
  Play,
  Copy,
  Check,
  Server,
  Layers,
  Send,
  Trash2
} from "lucide-react";

interface OllamaTerminalModalProps {
  isOpen: boolean;
  onClose: () => void;
}

export const OllamaTerminalModal: React.FC<OllamaTerminalModalProps> = ({ isOpen, onClose }) => {
  const [models, setModels] = useState<Array<any>>([]);
  const [loadingModels, setLoadingModels] = useState(false);

  // Deploy / Create Model State
  const [deployModelName, setDeployModelName] = useState("drjones-tool-beast");
  const [modelfileText, setModelfileText] = useState(
    `FROM ./deploy_infra_model/unsloth.Q4_K_M.gguf\nPARAMETER num_ctx 4096\nPARAMETER temperature 0.6\nPARAMETER top_p 0.9\nSYSTEM "You are an expert AI model variant fine-tuned for code generation, structured tool calling, and high-performance execution."`
  );
  const [deploying, setDeploying] = useState(false);
  const [deployMsg, setDeployMsg] = useState<string | null>(null);

  // CLI Terminal State
  const [terminalLogs, setTerminalLogs] = useState<Array<{ cmd: string; output: string }>>([
    { cmd: "ollama list", output: "NAME                      ID           SIZE     MODIFIED\ndrjones-tool-beast:latest c829fa11     6.2 GB   Just now\nllama3.1:8b-instruct-q4_0 e912aa21     4.7 GB   2 hours ago\nqwen2.5-coder:32b         a192bb44     19.8 GB  Yesterday" }
  ]);
  const [cliInput, setCliInput] = useState("ollama list");
  const [executing, setExecuting] = useState(false);
  const [copiedName, setCopiedName] = useState<string | null>(null);

  const termEndRef = useRef<HTMLDivElement>(null);

  // Fetch installed Ollama models
  const fetchModels = async () => {
    setLoadingModels(true);
    try {
      const res = await fetch("/api/ollama/models");
      const data = await res.json();
      if (data.models) {
        setModels(data.models);
      }
    } catch (e) {
      console.error("Failed to fetch models", e);
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    if (isOpen) {
      fetchModels();
    }
  }, [isOpen]);

  useEffect(() => {
    if (termEndRef.current) {
      termEndRef.current.scrollIntoView({ behavior: "smooth" });
    }
  }, [terminalLogs]);

  // Execute terminal command
  const handleExecCmd = async () => {
    if (!cliInput.trim()) return;
    const cmd = cliInput;
    setExecuting(true);

    try {
      const res = await fetch("/api/terminal/exec", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ command: cmd })
      });
      const data = await res.json();
      setTerminalLogs((prev) => [...prev, { cmd, output: data.stdout || data.error || "Done." }]);
    } catch (e: any) {
      setTerminalLogs((prev) => [...prev, { cmd, output: `Execution error: ${e.message}` }]);
    } finally {
      setExecuting(false);
      setCliInput("");
    }
  };

  // Upload / Create model in Ollama
  const handleUploadToOllama = async () => {
    setDeploying(true);
    setDeployMsg(null);

    try {
      const res = await fetch("/api/ollama/create", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          modelName: deployModelName,
          modelfileContent: modelfileText
        })
      });
      const data = await res.json();
      setDeployMsg(data.message || "Model uploaded & registered to Ollama successfully!");
      fetchModels();
    } catch (e: any) {
      setDeployMsg(`Upload error: ${e.message}`);
    } finally {
      setDeploying(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 bg-black/80 backdrop-blur-md flex items-center justify-center p-4">
      <div className="bg-[#121214] border border-[#27272a] max-w-5xl w-full rounded-2xl shadow-2xl flex flex-col max-h-[90vh] overflow-hidden">
        
        {/* Header */}
        <div className="px-6 py-4 border-b border-[#27272a] flex items-center justify-between bg-[#0c0c0e]">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 rounded-lg bg-blue-600/20 border border-blue-500/30 flex items-center justify-center text-blue-400">
              <TerminalIcon className="w-4 h-4" />
            </div>
            <div>
              <h2 className="text-base font-bold font-outfit text-white flex items-center gap-2">
                OLLAMA TERMINAL & LOCAL MODEL MANAGER
              </h2>
              <p className="text-xs text-zinc-400">Inspect installed models on machine, register custom GGUF models & run CLI commands</p>
            </div>
          </div>

          <button onClick={onClose} className="text-zinc-400 hover:text-white cursor-pointer">
            <X className="w-5 h-5" />
          </button>
        </div>

        {/* Content Body (2 Columns) */}
        <div className="flex-1 overflow-y-auto p-6 grid grid-cols-1 lg:grid-cols-12 gap-6 custom-scrollbar">

          {/* Left Column: Installed Models & Upload to Ollama (6 cols) */}
          <div className="lg:col-span-6 space-y-6">

            {/* Installed Models Section */}
            <div className="bg-zinc-950 border border-zinc-800 p-4 rounded-xl space-y-3">
              <div className="flex items-center justify-between">
                <h3 className="text-xs font-bold font-mono text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                  <Server className="w-4 h-4 text-emerald-400" /> INSTALLED OLLAMA MODELS (`ollama list`)
                </h3>
                <button
                  onClick={fetchModels}
                  disabled={loadingModels}
                  className="text-zinc-400 hover:text-white text-xs font-mono flex items-center gap-1 cursor-pointer"
                >
                  <RefreshCw className={`w-3.5 h-3.5 ${loadingModels ? "animate-spin" : ""}`} /> Refresh
                </button>
              </div>

              <div className="space-y-2 max-h-[200px] overflow-y-auto custom-scrollbar pr-1">
                {models.length === 0 ? (
                  <div className="text-xs text-zinc-500 italic py-4 text-center">No local Ollama models found.</div>
                ) : (
                  models.map((m, idx) => (
                    <div key={idx} className="bg-zinc-900/70 border border-zinc-800 p-3 rounded-lg flex items-center justify-between text-xs font-mono">
                      <div>
                        <div className="text-white font-bold flex items-center gap-2">
                          {m.name}
                          {(m.name.includes("drjones") || m.name.includes("tool-beast")) && (
                            <span className="px-1.5 py-0.2 bg-emerald-500/20 text-emerald-400 border border-emerald-500/30 text-[9px] rounded font-bold">YOUR MODEL</span>
                          )}
                        </div>
                        <div className="text-[10px] text-zinc-500 mt-0.5">
                          Size: {typeof m.size === "number" ? (m.size / (1024 * 1024 * 1024)).toFixed(1) + " GB" : m.size || "4.7 GB"}
                          {m.details?.parameter_size ? ` • ${m.details.parameter_size}` : ""}
                          {m.details?.quantization_level ? ` (${m.details.quantization_level})` : ""}
                        </div>
                      </div>
                      <button
                        onClick={() => {
                          navigator.clipboard.writeText(m.name);
                          setCopiedName(m.name);
                          setTimeout(() => setCopiedName(null), 2000);
                        }}
                        className="px-2.5 py-1 bg-zinc-800 hover:bg-zinc-700 text-zinc-300 rounded border border-zinc-700 flex items-center gap-1 cursor-pointer text-[11px]"
                      >
                        {copiedName === m.name ? <Check className="w-3 h-3 text-emerald-400" /> : <Copy className="w-3 h-3" />}
                        Copy
                      </button>
                    </div>
                  ))
                )}
              </div>
            </div>

            {/* Upload / Register Custom Model Section */}
            <div className="bg-zinc-950 border border-zinc-800 p-4 rounded-xl space-y-3">
              <h3 className="text-xs font-bold font-mono text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                <UploadCloud className="w-4 h-4 text-blue-400" /> REGISTER / UPLOAD MODEL TO OLLAMA
              </h3>

              <div className="space-y-3 text-xs font-mono">
                <div>
                  <label className="text-zinc-400 block mb-1">Target Ollama Model Name:</label>
                  <input
                    type="text"
                    value={deployModelName}
                    onChange={(e) => setDeployModelName(e.target.value)}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-white font-mono focus:outline-none focus:border-blue-500"
                  />
                </div>

                <div>
                  <label className="text-zinc-400 block mb-1">Modelfile Generator Script:</label>
                  <textarea
                    value={modelfileText}
                    onChange={(e) => setModelfileText(e.target.value)}
                    rows={4}
                    className="w-full bg-zinc-900 border border-zinc-800 rounded-lg p-3 text-xs font-mono text-zinc-300 focus:outline-none focus:border-blue-500 custom-scrollbar leading-relaxed"
                  />
                </div>

                {deployMsg && (
                  <div className="p-2.5 rounded-lg bg-emerald-950/40 border border-emerald-500/30 text-emerald-400 text-xs font-mono">
                    ✓ {deployMsg}
                  </div>
                )}

                <button
                  onClick={handleUploadToOllama}
                  disabled={deploying}
                  className={`w-full py-2.5 rounded-xl font-bold font-mono text-xs transition cursor-pointer flex items-center justify-center gap-2 shadow-lg ${
                    deploying
                      ? "bg-zinc-800 text-zinc-500 cursor-not-allowed"
                      : "bg-blue-600 hover:bg-blue-500 text-white shadow-blue-600/30"
                  }`}
                >
                  <UploadCloud className="w-4 h-4" />
                  {deploying ? "REGISTERING TO OLLAMA..." : "🚀 UPLOAD / REGISTER MODEL TO OLLAMA"}
                </button>
              </div>
            </div>

          </div>

          {/* Right Column: Interactive Cyber CLI Terminal (6 cols) */}
          <div className="lg:col-span-6 bg-zinc-950 border border-zinc-800 p-4 rounded-xl flex flex-col justify-between h-[500px]">
            <div>
              <div className="flex items-center justify-between border-b border-zinc-900 pb-2.5 mb-3">
                <h3 className="text-xs font-bold font-mono text-zinc-300 uppercase tracking-wider flex items-center gap-2">
                  <TerminalIcon className="w-4 h-4 text-cyan-400" /> OLLAMA CLI TERMINAL (`$ prompt`)
                </h3>
                <span className="text-[10px] font-mono text-cyan-400 bg-cyan-500/10 px-2 py-0.5 rounded border border-cyan-500/30">
                  Executable Shell
                </span>
              </div>

              {/* Terminal Stream Output */}
              <div className="h-[370px] overflow-y-auto custom-scrollbar space-y-3 font-mono text-xs text-zinc-300 pr-1">
                {terminalLogs.map((log, i) => (
                  <div key={i} className="space-y-1">
                    <div className="text-cyan-400 font-bold flex items-center gap-1.5">
                      <span className="text-emerald-400">$</span> {log.cmd}
                    </div>
                    <pre className="bg-zinc-900/80 p-3 rounded-lg border border-zinc-800/80 text-zinc-300 overflow-x-auto whitespace-pre-wrap leading-relaxed">
                      {log.output}
                    </pre>
                  </div>
                ))}
                <div ref={termEndRef} />
              </div>
            </div>

            {/* Terminal CLI Input Bar */}
            <div className="pt-3 border-t border-zinc-900 flex items-center gap-2 font-mono">
              <span className="text-emerald-400 font-bold text-sm">$</span>
              <input
                type="text"
                value={cliInput}
                onChange={(e) => setCliInput(e.target.value)}
                onKeyDown={(e) => e.key === "Enter" && handleExecCmd()}
                placeholder="Type command (e.g. 'ollama list', 'ollama ps', 'nvidia-smi')..."
                className="flex-1 bg-zinc-900 border border-zinc-800 rounded-lg px-3 py-1.5 text-xs text-white focus:outline-none focus:border-cyan-500"
              />
              <button
                onClick={handleExecCmd}
                disabled={executing}
                className="px-3 py-1.5 bg-cyan-600 hover:bg-cyan-500 text-white rounded-lg text-xs font-bold cursor-pointer transition flex items-center gap-1"
              >
                <Send className="w-3 h-3" /> Exec
              </button>
            </div>
          </div>

        </div>

        {/* Footer */}
        <div className="px-6 py-3 border-t border-[#27272a] bg-[#0c0c0e] flex justify-end">
          <button onClick={onClose} className="px-5 py-2 bg-zinc-900 hover:bg-zinc-800 text-zinc-300 text-xs rounded-xl font-mono cursor-pointer">
            Close Terminal
          </button>
        </div>

      </div>
    </div>
  );
};
