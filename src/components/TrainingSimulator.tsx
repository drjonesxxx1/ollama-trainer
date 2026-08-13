import React, { useEffect, useState, useRef } from "react";
import {
  PlayCircle,
  PauseCircle,
  RotateCcw,
  Zap,
  ArrowRight,
  TrendingDown,
  Terminal,
} from "lucide-react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
} from "recharts";
import { BaseModelInfo, TrainingHyperparameters, TrainingLogEntry } from "../types";

interface TrainingSimulatorProps {
  selectedModel: BaseModelInfo;
  hyperparameters: TrainingHyperparameters;
  onProceed: () => void;
}

export const TrainingSimulator: React.FC<TrainingSimulatorProps> = ({
  selectedModel,
  hyperparameters,
  onProceed,
}) => {
  const [isRunning, setIsRunning] = useState(false);
  const [currentStep, setCurrentStep] = useState(0);
  const totalSteps = 60;
  const [logs, setLogs] = useState<TrainingLogEntry[]>([]);
  const [sampleGenerations, setSampleGenerations] = useState<string[]>([]);
  const timerRef = useRef<any>(null);

  // Generate initial point
  useEffect(() => {
    if (logs.length === 0) {
      setLogs([
        {
          step: 0,
          epoch: 0,
          loss: 2.85,
          learningRate: hyperparameters.learning_rate * 0.1,
          gradNorm: 1.42,
          vramUsedGb: 11.2,
          tokensPerSec: 2450,
          timestamp: new Date().toLocaleTimeString(),
        },
      ]);
    }
  }, []);

  useEffect(() => {
    if (isRunning) {
      timerRef.current = setInterval(() => {
        setCurrentStep((prev) => {
          if (prev >= totalSteps) {
            setIsRunning(false);
            clearInterval(timerRef.current);
            return prev;
          }
          const nextStep = prev + 1;

          // Realistic loss decay with stochastic noise
          const progress = nextStep / totalSteps;
          const baseLoss = 2.85 * Math.exp(-progress * 2.8) + 0.35;
          const noise = (Math.random() - 0.5) * 0.08;
          const currentLoss = Number(Math.max(0.25, baseLoss + noise).toFixed(4));

          // Cosine learning rate
          const lr = Number(
            (
              hyperparameters.learning_rate *
              0.5 *
              (1 + Math.cos((Math.PI * nextStep) / totalSteps))
            ).toExponential(2)
          );

          const gradNorm = Number((0.85 + Math.random() * 0.4).toFixed(3));
          const vramUsedGb = Number((11.4 + Math.sin(nextStep * 0.3) * 0.4).toFixed(1));
          const tokensPerSec = Math.round(2600 + (Math.random() - 0.5) * 200);

          const newLog: TrainingLogEntry = {
            step: nextStep,
            epoch: Number(((nextStep / totalSteps) * hyperparameters.epochs).toFixed(2)),
            loss: currentLoss,
            learningRate: lr,
            gradNorm,
            vramUsedGb,
            tokensPerSec,
            timestamp: new Date().toLocaleTimeString(),
          };

          setLogs((prevLogs) => [...prevLogs, newLog]);

          // Sample token generation preview at milestones
          if (nextStep === 15) {
            setSampleGenerations((g) => [
              `[Step 15 Checkpoint] Prompt: "Call the filesystem read_file tool"\nModel Output: {"name": "read_file", "path": "src/App.tsx"} (Loss: ${currentLoss})`,
              ...g,
            ]);
          } else if (nextStep === 35) {
            setSampleGenerations((g) => [
              `[Step 35 Checkpoint] Prompt: "Query database for top 5 active users"\nModel Output: <tool_call>{"name": "execute_sql", "arguments": {"query": "SELECT * FROM users ORDER BY created_at DESC LIMIT 5;"}}</tool_call>\nFound 5 users.`,
              ...g,
            ]);
          } else if (nextStep === 60) {
            setSampleGenerations((g) => [
              `[Step 60 Final] High precision MCP multi-turn tool calling & Deep Reasoning aligned perfectly! (Final Loss: ${currentLoss})`,
              ...g,
            ]);
          }

          return nextStep;
        });
      }, 400);
    } else {
      if (timerRef.current) clearInterval(timerRef.current);
    }
    return () => {
      if (timerRef.current) clearInterval(timerRef.current);
    };
  }, [isRunning, hyperparameters]);

  const handleReset = () => {
    setIsRunning(false);
    setCurrentStep(0);
    setLogs([
      {
        step: 0,
        epoch: 0,
        loss: 2.85,
        learningRate: hyperparameters.learning_rate * 0.1,
        gradNorm: 1.42,
        vramUsedGb: 11.2,
        tokensPerSec: 2450,
        timestamp: new Date().toLocaleTimeString(),
      },
    ]);
    setSampleGenerations([]);
  };

  const latestLog = logs[logs.length - 1] || logs[0];
  const progressPercent = Math.round((currentStep / totalSteps) * 100);

  return (
    <div className="space-y-6">
      {/* Top Banner & Control HUD */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 flex flex-wrap items-center justify-between gap-4">
        <div>
          <div className="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded text-[10px] font-mono text-blue-400 bg-blue-400/10 border border-blue-400/20 mb-2">
            <Zap className="w-3.5 h-3.5" /> UNSLOTH CUDA TRAINING ENGINE
          </div>
          <h2 className="text-lg font-semibold text-[#f4f4f5]">
            Live Fine-Tuning Execution & Telemetry Monitor
          </h2>
          <p className="text-xs text-zinc-400 mt-1">
            Running Triton backprop kernel on NVIDIA RTX 4080 Super with FlashAttention-2.
          </p>
        </div>

        {/* Action Controls */}
        <div className="flex items-center gap-2">
          {!isRunning ? (
            <button
              onClick={() => setIsRunning(true)}
              disabled={currentStep >= totalSteps}
              className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-emerald-600 hover:bg-emerald-500 text-white shadow-sm cursor-pointer disabled:opacity-50"
            >
              <PlayCircle className="w-4 h-4" />
              <span>{currentStep === 0 ? "Start Training Run" : "Resume Training"}</span>
            </button>
          ) : (
            <button
              onClick={() => setIsRunning(false)}
              className="flex items-center gap-2 px-4 py-2 rounded text-xs font-medium bg-amber-600 hover:bg-amber-500 text-white shadow-sm cursor-pointer"
            >
              <PauseCircle className="w-4 h-4" />
              <span>Pause Training</span>
            </button>
          )}

          <button
            onClick={handleReset}
            className="p-2 rounded bg-zinc-900 hover:bg-zinc-800 text-zinc-400 border border-zinc-800 cursor-pointer"
            title="Reset training simulation"
          >
            <RotateCcw className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Progress Bar & Telemetry Strip */}
      <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-3">
        <div className="flex justify-between items-center text-xs font-mono">
          <span className="text-zinc-300">
            Training Progress: Step {currentStep} of {totalSteps} ({progressPercent}%)
          </span>
          <span className="text-blue-400">
            Epoch {((currentStep / totalSteps) * hyperparameters.epochs).toFixed(2)} / {hyperparameters.epochs}
          </span>
        </div>
        <div className="w-full h-2 bg-black/40 rounded-full overflow-hidden border border-zinc-800">
          <div
            className="h-full bg-blue-600 rounded-full transition-all duration-300"
            style={{ width: `${progressPercent}%` }}
          />
        </div>

        {/* Real-time metrics grid */}
        <div className="grid grid-cols-2 md:grid-cols-5 gap-3 pt-2 text-xs">
          <div className="bg-black/30 p-3 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Current Loss</div>
            <div className="text-sm font-mono font-bold text-blue-400 mt-0.5">
              {latestLog?.loss ?? "--"}
            </div>
          </div>
          <div className="bg-black/30 p-3 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Learning Rate</div>
            <div className="text-sm font-mono font-bold text-zinc-200 mt-0.5">
              {latestLog?.learningRate ?? "--"}
            </div>
          </div>
          <div className="bg-black/30 p-3 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Grad Norm</div>
            <div className="text-sm font-mono font-bold text-zinc-200 mt-0.5">
              {latestLog?.gradNorm ?? "--"}
            </div>
          </div>
          <div className="bg-black/30 p-3 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">RTX 4080 VRAM</div>
            <div className="text-sm font-mono font-bold text-emerald-400 mt-0.5">
              {latestLog?.vramUsedGb} GB <span className="text-[10px] text-zinc-500 font-normal">/ 16GB</span>
            </div>
          </div>
          <div className="bg-black/30 p-3 rounded border border-zinc-800">
            <div className="text-[10px] font-mono text-zinc-500 uppercase">Throughput</div>
            <div className="text-sm font-mono font-bold text-cyan-400 mt-0.5">
              {latestLog?.tokensPerSec} tok/s
            </div>
          </div>
        </div>
      </div>

      {/* Loss Convergence Chart & Generation Checkpoint Logs */}
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Left: Recharts Live Loss Curve */}
        <div className="lg:col-span-2 bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-3">
          <div className="flex items-center justify-between">
            <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2">
              <TrendingDown className="w-4 h-4 text-blue-400" /> Training Loss Convergence Curve
            </h3>
            <span className="text-[10px] text-zinc-500 font-mono">Cross Entropy Loss (SFT)</span>
          </div>

          <div className="h-64 w-full bg-zinc-950 rounded p-2 border border-zinc-800">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={logs}>
                <CartesianGrid strokeDasharray="3 3" stroke="#27272a" />
                <XAxis dataKey="step" stroke="#71717a" fontSize={11} />
                <YAxis domain={["auto", "auto"]} stroke="#71717a" fontSize={11} />
                <Tooltip
                  contentStyle={{
                    backgroundColor: "#18181b",
                    borderColor: "#27272a",
                    fontSize: "11px",
                    borderRadius: "6px",
                    color: "#f4f4f5",
                  }}
                />
                <Line
                  type="monotone"
                  dataKey="loss"
                  stroke="#3b82f6"
                  strokeWidth={2}
                  dot={false}
                  isAnimationActive={false}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Right: Intermediate Checkpoint Samples */}
        <div className="bg-[#18181b]/50 border border-[#27272a] rounded-xl p-5 space-y-3 flex flex-col justify-between">
          <div>
            <h3 className="text-sm font-semibold text-[#f4f4f5] flex items-center gap-2 mb-3">
              <Terminal className="w-4 h-4 text-emerald-400" /> Checkpoint Generations
            </h3>

            <div className="space-y-2.5 max-h-60 overflow-y-auto">
              {sampleGenerations.length === 0 ? (
                <div className="text-xs text-zinc-500 italic p-3 bg-zinc-950 rounded border border-zinc-800">
                  Model checkpoint test outputs will appear here at steps 15, 35, and 60...
                </div>
              ) : (
                sampleGenerations.map((gen, idx) => (
                  <div
                    key={idx}
                    className="p-3 bg-zinc-950 rounded border border-zinc-800 text-[11px] font-mono text-emerald-400 whitespace-pre-wrap leading-relaxed"
                  >
                    {gen}
                  </div>
                ))
              )}
            </div>
          </div>

          <div className="pt-3 border-t border-[#27272a]">
            <button
              onClick={onProceed}
              className="w-full flex items-center justify-center gap-2 py-2 rounded font-medium text-xs bg-blue-600 hover:bg-blue-500 text-white shadow-sm cursor-pointer"
            >
              <span>Export Modelfile & Push to Ollama</span>
              <ArrowRight className="w-3.5 h-3.5" />
            </button>
          </div>
        </div>
      </div>
    </div>
  );
};
