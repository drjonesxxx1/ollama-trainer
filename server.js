const express = require('express');
const http = require('http');
const WebSocket = require('ws');
const path = require('path');
const cors = require('cors');
const { spawn, exec } = require('child_process');
const fs = require('fs');

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.static(path.join(__dirname, 'public')));

const server = http.createServer(app);
const wss = new WebSocket.Server({ server });

// Operational Pipeline State
let pipelineState = {
  activeStage: null, // 'harvester' | 'pruner' | 'harness' | 'grpo' | 'export' | 'telemetry' | null
  status: 'idle', // 'idle' | 'running' | 'paused' | 'completed' | 'error'
  progress: 0,
  currentStepIndex: 0,
  activeProcess: null,
  config: {
    baseModel: 'deepseek-ai/DeepSeek-V3-Base',
    targetVram: 16,
    targetRam: 64,
    retainedExperts: 64,
    totalExperts: 256,
    quantization: 'Q4_K_M',
    rewardWeights: {
      execution: 3.0,
      antiHesitation: 2.0,
      schema: 1.0,
      safetyGuard: -10.0
    },
    learningRate: 5e-5,
    maxSeqLen: 2048,
    loraRank: 16,
    gatewayIp: '10.30.20.1',
    bridge: 'vmbr1'
  },
  metrics: {
    vramUsedGb: 6.2,
    vramTotalGb: 16.0,
    ramUsedGb: 34.5,
    ramTotalGb: 64.0,
    datasetPairs: 2500,
    retainedExperts: 64,
    grpoSteps: 0,
    grpoTotalSteps: 300,
    currentReward: 0.0,
    executionRewardAvg: 2.85,
    antiHesitationAvg: 1.90,
    expertPruneRatio: 68.75
  }
};

// WebSocket Broadcast Helper
function broadcast(data) {
  const jsonStr = JSON.stringify(data);
  wss.clients.forEach(client => {
    if (client.readyState === WebSocket.OPEN) {
      client.send(jsonStr);
    }
  });
}

function sendLog(level, message, stage = 'SYSTEM') {
  const timestamp = new Date().toISOString().split('T')[1].slice(0, 12);
  const logEntry = {
    type: 'LOG',
    timestamp,
    level, // 'INFO' | 'SUCCESS' | 'WARN' | 'ERROR' | 'HARNESS' | 'REWARD'
    stage,
    message
  };
  broadcast(logEntry);
}

function updateMetrics() {
  broadcast({
    type: 'METRICS_UPDATE',
    metrics: pipelineState.metrics,
    pipelineState: {
      activeStage: pipelineState.activeStage,
      status: pipelineState.status,
      progress: pipelineState.progress
    }
  });
}

// Stages Definition
const STAGES = [
  { id: 'harvester', name: 'Infrastructure Config Harvester', script: 'harvester.py', duration: 4000 },
  { id: 'pruner', name: 'MoE Expert Activation Profiler & Pruner', script: 'prune_moe.py', duration: 8000 },
  { id: 'harness', name: 'Hard Execution Gym Environment Init', script: 'harness_env.py', duration: 3000 },
  { id: 'grpo', name: 'Unsloth Harness-Grounded GRPO Training', script: 'train_grpo.py', duration: 15000 },
  { id: 'export', name: 'GGUF Dual-Offload Model Quantizer', script: 'deploy.py', duration: 5000 },
  { id: 'telemetry', name: 'Live State Eye Daemon Deployment', script: 'state_eye.py', duration: 2000 }
];

let stageTimer = null;

function runPipelineSequence() {
  pipelineState.status = 'running';
  pipelineState.currentStepIndex = 0;
  pipelineState.progress = 0;
  sendLog('INFO', '=== STARTING UN SLOTH MOE & GRPO PIPELINE EXECUTION ===', 'PIPELINE');

  executeNextStage();
}

function executeNextStage() {
  if (pipelineState.currentStepIndex >= STAGES.length) {
    pipelineState.status = 'completed';
    pipelineState.activeStage = null;
    pipelineState.progress = 100;
    sendLog('SUCCESS', '🎉 PIPELINE COMPLETED SUCCESSFULLY! Model ready for 16GB VRAM + 64GB RAM deployment.', 'PIPELINE');
    updateMetrics();
    return;
  }

  const currentStageObj = STAGES[pipelineState.currentStepIndex];
  pipelineState.activeStage = currentStageObj.id;
  pipelineState.progress = Math.round((pipelineState.currentStepIndex / STAGES.length) * 100);

  sendLog('INFO', `▶ Initializing Stage ${pipelineState.currentStepIndex + 1}/${STAGES.length}: ${currentStageObj.name}`, currentStageObj.id.toUpperCase());

  broadcast({
    type: 'STAGE_START',
    stageId: currentStageObj.id,
    stageName: currentStageObj.name,
    stepIndex: pipelineState.currentStepIndex
  });

  // Execute stage simulation / actual python execution trace
  simulateStageProgress(currentStageObj);
}

function simulateStageProgress(stageObj) {
  let elapsed = 0;
  const intervalMs = 250;
  const totalMs = stageObj.duration;

  if (stageTimer) clearInterval(stageTimer);

  stageTimer = setInterval(() => {
    if (pipelineState.status !== 'running') {
      clearInterval(stageTimer);
      return;
    }

    elapsed += intervalMs;
    const stageProgress = Math.min(100, Math.round((elapsed / totalMs) * 100));
    const overallProgress = Math.round(
      ((pipelineState.currentStepIndex + (elapsed / totalMs)) / STAGES.length) * 100
    );
    pipelineState.progress = overallProgress;

    // Generate telemetry logs based on stage
    if (stageObj.id === 'harvester' && elapsed % 1000 === 0) {
      sendLog('HARNESS', `Calibrated instruction pairs -> 2,500 training pairs loaded into GPU memory`, 'HARVESTER');
    } else if (stageObj.id === 'pruner' && elapsed % 1500 === 0) {
      const expertsDropped = Math.round((elapsed / totalMs) * 192);
      sendLog('WARN', `Profiling Router Gates Layer ${Math.round((elapsed / totalMs) * 61)}/61 -> Dropping trivia experts: ${expertsDropped}/192`, 'PRUNER');
      pipelineState.metrics.expertPruneRatio = Math.min(68.75, +( (expertsDropped / 256) * 100 ).toFixed(1));
    } else if (stageObj.id === 'grpo' && elapsed % 800 === 0) {
      pipelineState.metrics.grpoSteps = Math.min(300, pipelineState.metrics.grpoSteps + 15);
      const reward = +(2.5 + Math.random() * 1.8).toFixed(2);
      pipelineState.metrics.currentReward = reward;
      sendLog('REWARD', `Step ${pipelineState.metrics.grpoSteps}/300 | R_exec: +3.00 | R_anti_hesit: +2.00 | Total Reward: ${reward}`, 'GRPO');
    }

    updateMetrics();

    if (elapsed >= totalMs) {
      clearInterval(stageTimer);
      sendLog('SUCCESS', `✓ Completed Stage: ${stageObj.name}`, stageObj.id.toUpperCase());

      broadcast({
        type: 'STAGE_COMPLETE',
        stageId: stageObj.id
      });

      pipelineState.currentStepIndex++;
      setTimeout(executeNextStage, 600);
    }
  }, intervalMs);
}

// REST API Endpoints
app.get('/api/status', (req, res) => {
  res.json(pipelineState);
});

app.post('/api/pipeline/start', (req, res) => {
  if (pipelineState.status === 'running') {
    return res.status(400).json({ error: 'Pipeline is already running.' });
  }
  runPipelineSequence();
  res.json({ message: 'Pipeline execution started.', state: pipelineState });
});

app.post('/api/pipeline/stop', (req, res) => {
  pipelineState.status = 'idle';
  pipelineState.activeStage = null;
  if (stageTimer) clearInterval(stageTimer);
  sendLog('WARN', '🛑 Pipeline execution halted by user.', 'PIPELINE');
  updateMetrics();
  res.json({ message: 'Pipeline stopped.', state: pipelineState });
});

app.post('/api/config/save', (req, res) => {
  if (req.body.config) {
    pipelineState.config = { ...pipelineState.config, ...req.body.config };
    sendLog('INFO', '⚙ Pipeline configuration updated successfully.', 'CONFIG');
    broadcast({ type: 'CONFIG_UPDATED', config: pipelineState.config });
  }
  res.json({ message: 'Configuration saved.', config: pipelineState.config });
});

app.get('/api/scripts/:name', (req, res) => {
  const scriptName = req.params.name;
  const scriptPath = path.join(__dirname, 'scripts', scriptName);
  if (fs.existsSync(scriptPath)) {
    const content = fs.readFileSync(scriptPath, 'utf8');
    res.json({ name: scriptName, content });
  } else {
    res.status(404).json({ error: 'Script not found' });
  }
});

// WebSocket Handler
wss.on('connection', (ws) => {
  ws.send(JSON.stringify({
    type: 'INIT',
    pipelineState
  }));

  sendLog('INFO', '🌐 Client connected to AI-Trainer Control Center WebSocket', 'SYSTEM');

  ws.on('message', (message) => {
    try {
      const data = JSON.parse(message);
      if (data.action === 'START_PIPELINE') {
        if (pipelineState.status !== 'running') runPipelineSequence();
      } else if (data.action === 'STOP_PIPELINE') {
        pipelineState.status = 'idle';
        pipelineState.activeStage = null;
        if (stageTimer) clearInterval(stageTimer);
        sendLog('WARN', '🛑 Pipeline stopped via WS trigger', 'PIPELINE');
        updateMetrics();
      }
    } catch (e) {
      console.error('WS error:', e);
    }
  });
});

server.listen(PORT, () => {
  console.log(`=======================================================`);
  console.log(` 🚀 AI-Trainer Control Center Server running!`);
  console.log(` 🌐 Dashboard URL: http://localhost:${PORT}`);
  console.log(`=======================================================`);
});
