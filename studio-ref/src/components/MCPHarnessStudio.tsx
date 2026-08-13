import React, { useState } from "react";
import {
  Wrench,
  Plus,
  Trash2,
  Play,
  Sparkles,
  Code2,
  Terminal,
  ArrowRight,
  RefreshCw,
} from "lucide-react";
import { MCPToolDeclaration, TrainingDataSample } from "../types";

interface MCPHarnessStudioProps {
  mcpTools: MCPToolDeclaration[];
  setMcpTools: React.Dispatch<React.SetStateAction<MCPToolDeclaration[]>>;
  dataset: TrainingDataSample[];
  setDataset: React.Dispatch<React.SetStateAction<TrainingDataSample[]>>;
  onProceed: () => void;
}

export const MCPHarnessStudio: React.FC<MCPHarnessStudioProps> = ({
  mcpTools,
  setMcpTools,
  dataset,
  setDataset,
  onProceed,
}) => {
  const [selectedToolId, setSelectedToolId] = useState<string>(mcpTools[0]?.id || "filesystem-mcp");
  const [generatingMcpPairs, setGeneratingMcpPairs] = useState(false);
  const [testUserPrompt, setTestUserPrompt] = useState("Read package.json and summarize our frontend dependencies.");
  const [harnessOutput, setHarnessOutput] = useState<string | null>(null);

  // New tool creator state
  const [newToolName, setNewToolName] = useState("");
  const [newServerName, setNewServerName] = useState("");
  const [newToolDesc, setNewToolDesc] = useState("");
  const [newToolSchemaJson, setNewToolSchemaJson] = useState(`{
  "type": "object",
  "properties": {
    "query": { "type": "string", "description": "Search query or input parameter" }
  },
  "required": ["query"]
}`);

  const activeTool = mcpTools.find((t) => t.id === selectedToolId) || mcpTools[0];

  const handleAddTool = () => {
    if (!newToolName.trim()) return;
    try {
      const parsedSchema = JSON.parse(newToolSchemaJson);
      const created: MCPToolDeclaration = {
        id: `tool-${Date.now()}`,
        name: newToolName.trim(),
        serverName: newServerName.trim() || "Custom MCP",
        description: newToolDesc.trim() || "Custom tool execution",
        parametersSchema: parsedSchema,
        sampleCallsCount: 0,
      };
      setMcpTools((prev) => [...prev, created]);
      setSelectedToolId(created.id);
      setNewToolName("");
      setNewToolDesc("");
    } catch (e: any) {
      console.error(e);
    }
  };

  const handleDeleteTool = (id: string) => {
    setMcpTools((prev) => prev.filter((t) => t.id !== id));
    if (selectedToolId === id && mcpTools.length > 1) {
      setSelectedToolId(mcpTools.find((t) => t.id !== id)!.id);
    }
  };

  const handleGenerateMCPDataWithAI = async () => {
    setGeneratingMcpPairs(true);
    try {
      const res = await fetch("/api/dataset/generate-mcp", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          mcpServers: mcpTools,
          count: 6,
          complexity: "advanced",
        }),
      });
      const data = await res.json();
      if (data.success && Array.isArray(data.data)) {
        const newSamples: TrainingDataSample[] = data.data.map((item: any) => ({
          id: `mcp-${Date.now()}-${Math.random().toString(36).substring(2, 6)}`,
          instruction: item.userQuery,
          output: `<tool_call>\n${JSON.stringify(item.toolCalls?.[0] || {}, null, 2)}\n</tool_call>\n\n${item.assistantResponse}`,
          category: "MCP Plugin",
          difficulty: "Hard",
          isMcpSample: true,
        }));
        setDataset((prev) => [...newSamples, ...prev]);
      }
    } catch (e: any) {
      console.error(e);
    } finally {
      setGeneratingMcpPairs(false);
    }
  };

  const handleRunHarnessSimulation = () => {
    setHarnessOutput("Simulating MCP execution harness...");
    setTimeout(() => {
      setHarnessOutput(`[MCP HARNESS] Matched Tool: ${activeTool.name} (${activeTool.serverName})
[PAYLOAD EMITTED] { "path": "package.json" }
[MCP RESPONSE] Status: 200 OK (Read 36 lines)
[MODEL SYNTHESIS] The application contains React 19, Vite, Express, and @google/genai as core dependencies.`);
    }, 600);
  };

  return (
    <div className="space-y-6">
      {/* Header Banner */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div className="max-w-2xl">
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-blue-400 bg-blue-400/10 border border-blue-400/20 mb-2">
            <Wrench className="w-3.5 h-3.5 text-blue-400" /> MCP (MODEL CONTEXT PROTOCOL) HARNESS
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Train Local Models for Flawless MCP Plugin Execution
          </h2>
          <p className="text-xs text-zinc-400 mt-1 leading-relaxed">
            Configure MCP server tool definitions (Filesystem, Postgres, Web Search, Terminal, GitHub) and auto-generate multi-turn function call datasets so your fine-tuned Ollama model executes tools in Cline, Cursor, Windsurf, and Claude Desktop.
          </p>
        </div>

        <button
          onClick={handleGenerateMCPDataWithAI}
          disabled={generatingMcpPairs}
          className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer disabled:opacity-50"
        >
          {generatingMcpPairs ? (
            <>
              <RefreshCw className="w-3.5 h-3.5 animate-spin" />
              <span>Generating MCP Pairs...</span>
            </>
          ) : (
            <>
              <Sparkles className="w-3.5 h-3.5" />
              <span>Auto-Synthesize Tool Pairs (AI)</span>
            </>
          )}
        </button>
      </div>

      {/* Main Grid: Tool Registry + Schema & Harness Playground */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Registered MCP Tools */}
        <div className="space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-[10px] font-mono font-bold text-zinc-500 uppercase tracking-wider">
              Active MCP Tools ({mcpTools.length})
            </h3>
            <span className="text-[10px] font-mono text-blue-400 bg-blue-400/10 px-1.5 py-0.5 rounded border border-blue-400/20">
              JSON SCHEMA
            </span>
          </div>

          <div className="space-y-1.5 max-h-[500px] overflow-y-auto pr-1">
            {mcpTools.map((tool) => {
              const isSelected = selectedToolId === tool.id;
              return (
                <div
                  key={tool.id}
                  onClick={() => setSelectedToolId(tool.id)}
                  className={`p-3 rounded-lg border transition-all cursor-pointer ${
                    isSelected
                      ? "bg-[#18181b] border-blue-500/40 text-blue-400 ring-1 ring-blue-500/20"
                      : "bg-[#121214] border-[#27272a] hover:border-zinc-700 hover:bg-[#18181b]/40 text-zinc-300"
                  }`}
                >
                  <div className="flex items-center justify-between">
                    <span className="font-mono text-xs font-semibold text-[#f4f4f5]">{tool.name}</span>
                    <span className="text-[10px] font-mono text-zinc-400 bg-zinc-900 px-1.5 py-0.5 rounded border border-zinc-800">
                      {tool.serverName}
                    </span>
                  </div>
                  <p className="text-[11px] text-zinc-400 mt-1 line-clamp-1">{tool.description}</p>
                </div>
              );
            })}
          </div>

          {/* Add New Tool Card */}
          <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-4 space-y-2.5 text-xs">
            <div className="font-semibold text-[#f4f4f5] flex items-center gap-1.5 text-xs">
              <Plus className="w-3.5 h-3.5 text-blue-400" /> Declare New MCP Tool
            </div>
            <input
              type="text"
              placeholder="Tool Name (e.g. docker_container_exec)"
              value={newToolName}
              onChange={(e) => setNewToolName(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <input
              type="text"
              placeholder="Server (e.g. Docker MCP Server)"
              value={newServerName}
              onChange={(e) => setNewServerName(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <input
              type="text"
              placeholder="Description for model prompt"
              value={newToolDesc}
              onChange={(e) => setNewToolDesc(e.target.value)}
              className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 placeholder-zinc-600 focus:ring-1 focus:ring-blue-600 outline-none"
            />
            <button
              onClick={handleAddTool}
              className="w-full py-1.5 rounded text-xs font-medium bg-zinc-900 hover:bg-zinc-800 text-zinc-300 border border-zinc-800 cursor-pointer"
            >
              Add Tool to Training Pipeline
            </button>
          </div>
        </div>

        {/* Right: Active Tool Schema & Test Harness Simulator */}
        <div className="lg:col-span-2 space-y-6">
          {activeTool && (
            <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-4">
              <div className="flex items-center justify-between border-b border-[#27272a] pb-3">
                <div>
                  <div className="text-[10px] text-blue-400 font-mono uppercase">{activeTool.serverName}</div>
                  <h3 className="text-base font-semibold text-[#f4f4f5] font-mono">{activeTool.name}</h3>
                </div>
                <button
                  onClick={() => handleDeleteTool(activeTool.id)}
                  className="text-zinc-500 hover:text-rose-400 p-1.5 rounded border border-transparent hover:border-zinc-700 cursor-pointer"
                >
                  <Trash2 className="w-3.5 h-3.5" />
                </button>
              </div>

              <div className="text-xs text-zinc-400 leading-relaxed">
                <span className="font-medium text-zinc-300 block mb-1">Docstring / Instructions:</span>
                {activeTool.description}
              </div>

              {/* JSON Schema */}
              <div>
                <div className="text-xs font-medium text-zinc-400 mb-1.5 flex items-center gap-1.5">
                  <Code2 className="w-3.5 h-3.5 text-blue-400" /> Parameter JSON Schema
                </div>
                <pre className="bg-zinc-950 p-3 rounded text-xs font-mono text-zinc-300 overflow-x-auto border border-zinc-800">
                  {JSON.stringify(activeTool.parametersSchema, null, 2)}
                </pre>
              </div>

              {/* Interactive MCP Test Harness Simulator */}
              <div className="bg-black/30 p-4 rounded border border-zinc-800 space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium text-[#f4f4f5] flex items-center gap-1.5">
                    <Terminal className="w-3.5 h-3.5 text-blue-400" /> Interactive Harness Verification
                  </span>
                  <button
                    onClick={handleRunHarnessSimulation}
                    className="flex items-center gap-1 px-3 py-1 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white cursor-pointer shadow-sm"
                  >
                    <Play className="w-3 h-3" /> Test Trigger
                  </button>
                </div>

                <input
                  type="text"
                  value={testUserPrompt}
                  onChange={(e) => setTestUserPrompt(e.target.value)}
                  className="w-full bg-zinc-950 border border-zinc-800 rounded px-2.5 py-1.5 text-xs text-zinc-200 focus:ring-1 focus:ring-blue-600 outline-none font-mono"
                />

                {harnessOutput && (
                  <pre className="p-3 bg-zinc-950 rounded text-[11px] font-mono text-emerald-400 border border-zinc-800 whitespace-pre-wrap leading-relaxed">
                    {harnessOutput}
                  </pre>
                )}
              </div>
            </div>
          )}

          <div className="flex justify-end">
            <button
              onClick={onProceed}
              className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-blue-600 hover:bg-blue-500 text-white shadow-sm transition-all cursor-pointer"
            >
              <span>Proceed to Model-to-Model Distillation</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
