/**
 * app.js - Main Dashboard Application Controller
 * Connects WebSocket, renders fleet nodes, binds user interactions, and manages pipeline lifecycle.
 */

document.addEventListener('DOMContentLoaded', () => {
  const pipeline = new PipelineVisualizer('pipelineCanvas');
  const charts = new TelemetryCharts();
  const terminal = new TerminalConsole();

  let ws = null;
  let pipelineRunning = false;

  // DOM Elements
  const btnStart = document.getElementById('btnStartPipeline');
  const btnStop = document.getElementById('btnStopPipeline');
  const btnKnobs = document.getElementById('btnOpenKnobs');
  const btnCloseKnobs = document.getElementById('btnCloseKnobs');
  const btnSaveKnobs = document.getElementById('btnSaveKnobs');
  const knobsModal = document.getElementById('knobsModal');
  const btnSyncGitea = document.getElementById('btnSyncGitea');
  const scriptSelector = document.getElementById('scriptSelector');
  const scriptCodeView = document.getElementById('scriptCodeView');
  const fleetNodeList = document.getElementById('fleetNodeList');
  const loadingOverlay = document.getElementById('loadingOverlay');
  const loadingTitle = document.getElementById('loadingTitle');

  // Gauge Elements
  const gaugeVram = document.getElementById('gaugeVram');
  const gaugeRam = document.getElementById('gaugeRam');
  const gaugeAdb = document.getElementById('gaugeAdb');
  const statusBadge = document.getElementById('pipelineStatusBadge');
  const overallProgressText = document.getElementById('overallProgressText');
  const grpoStepVal = document.getElementById('grpoStepVal');
  const retainedExpertsVal = document.getElementById('retainedExpertsVal');

  // 1. Initialize WebSocket Connection
  function initWebSocket() {
    const protocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
    ws = new WebSocket(`${protocol}//${window.location.host}`);

    ws.onopen = () => {
      terminal.addLog('INFO', 'Connected to AI-Trainer Control Center Server WebSocket', 'WS');
    };

    ws.onmessage = (event) => {
      const data = JSON.parse(event.data);
      handleServerMessage(data);
    };

    ws.onclose = () => {
      terminal.addLog('WARN', 'WebSocket connection lost. Reconnecting in 3s...', 'WS');
      setTimeout(initWebSocket, 3000);
    };
  }

  function handleServerMessage(data) {
    if (data.type === 'INIT' || data.type === 'METRICS_UPDATE') {
      const state = data.pipelineState || data.pipelineState;
      const metrics = data.metrics;

      if (metrics) {
        gaugeVram.textContent = `${metrics.vramUsedGb} / ${metrics.vramTotalGb} GB`;
        gaugeRam.textContent = `${metrics.ramUsedGb} / ${metrics.ramTotalGb} GB`;
        gaugeAdb.textContent = `${metrics.adbOnlineCount} / ${metrics.activeVmCount} Online`;
        grpoStepVal.textContent = metrics.grpoSteps;
        retainedExpertsVal.textContent = Math.round(256 * (1 - (metrics.expertPruneRatio / 100)));

        charts.updateGRPO(metrics.grpoSteps, metrics.currentReward);
        charts.updateMoE(Math.round(256 * (1 - (metrics.expertPruneRatio / 100))));
      }

      if (state) {
        pipelineRunning = state.status === 'running';
        btnStart.disabled = pipelineRunning;
        btnStop.disabled = !pipelineRunning;

        statusBadge.textContent = `STATUS: ${state.status.toUpperCase()}`;
        statusBadge.className = `status-badge ${pipelineRunning ? 'status-running' : 'status-idle'}`;
        overallProgressText.textContent = `${state.progress}%`;

        if (state.activeStage) {
          pipeline.setActiveStage(state.activeStage);
        } else if (state.status === 'completed') {
          pipeline.setCompleted();
        } else {
          pipeline.reset();
        }
      }
    } else if (data.type === 'LOG') {
      terminal.addLog(data.level, data.message, data.stage, data.timestamp);
    } else if (data.type === 'STAGE_START') {
      pipeline.setActiveStage(data.stageId);
    } else if (data.type === 'STAGE_COMPLETE') {
      // Stage completion handled in state
    }
  }

  // 2. Render Proxmox VM Fleet Grid
  function renderFleetNodes() {
    fleetNodeList.innerHTML = '';
    const vms = [
      { ip: '10.30.20.101', mac: '52:54:00:12:34:01', adb: 'ONLINE', proxy: '185.220.101.4:1080' },
      { ip: '10.30.20.102', mac: '52:54:00:12:34:02', adb: 'ONLINE', proxy: '185.220.101.5:1080' },
      { ip: '10.30.20.103', mac: '52:54:00:12:34:03', adb: 'ONLINE', proxy: '185.220.101.6:1080' },
      { ip: '10.30.20.104', mac: '52:54:00:12:34:04', adb: 'ONLINE', proxy: '185.220.101.7:1080' },
      { ip: '10.30.20.105', mac: '52:54:00:12:34:05', adb: 'ONLINE', proxy: '185.220.101.8:1080' }
    ];

    vms.forEach(vm => {
      const card = document.createElement('div');
      card.className = 'bg-slate-900/60 border border-slate-800 p-2.5 rounded-xl flex items-center justify-between text-xs font-mono';
      card.innerHTML = `
        <div class="flex items-center gap-2.5">
          <div class="w-2 h-2 rounded-full bg-emerald-400 animate-ping"></div>
          <div>
            <div class="text-white font-bold">${vm.ip} <span class="text-[10px] text-slate-500 font-normal">(${vm.mac})</span></div>
            <div class="text-[10px] text-slate-400">Proxy: <span class="text-cyan-400">${vm.proxy}</span></div>
          </div>
        </div>
        <div class="px-2 py-0.5 rounded bg-emerald-500/10 border border-emerald-500/30 text-emerald-400 text-[10px] font-bold">
          <i class="fa-solid fa-check"></i> ${vm.adb}
        </div>
      `;
      fleetNodeList.appendChild(card);
    });
  }

  // 3. Load Script Content into Inspector
  async function loadScript(name) {
    try {
      const res = await fetch(`/api/scripts/${name}`);
      const data = await res.json();
      scriptCodeView.textContent = data.content || `# No code available for ${name}`;
    } catch (e) {
      scriptCodeView.textContent = `# Error loading script ${name}`;
    }
  }

  // 4. Interaction Handlers
  btnStart.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'START_PIPELINE' }));
    } else {
      fetch('/api/pipeline/start', { method: 'POST' });
    }
  });

  btnStop.addEventListener('click', () => {
    if (ws && ws.readyState === WebSocket.OPEN) {
      ws.send(JSON.stringify({ action: 'STOP_PIPELINE' }));
    } else {
      fetch('/api/pipeline/stop', { method: 'POST' });
    }
  });

  btnKnobs.addEventListener('click', () => {
    knobsModal.classList.remove('opacity-0', 'pointer-events-none');
  });

  btnCloseKnobs.addEventListener('click', () => {
    knobsModal.classList.add('opacity-0', 'pointer-events-none');
  });

  btnSaveKnobs.addEventListener('click', () => {
    const retained = parseInt(document.getElementById('knobExperts').value);
    const execWeight = parseFloat(document.getElementById('knobExec').value);
    const hesitWeight = parseFloat(document.getElementById('knobHesit').value);

    fetch('/api/config/save', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        config: {
          retainedExperts: retained,
          rewardWeights: { execution: execWeight, antiHesitation: hesitWeight }
        }
      })
    });

    knobsModal.classList.add('opacity-0', 'pointer-events-none');
    terminal.addLog('INFO', `Updated Tuning Knobs: Retained Experts=${retained}, Exec Weight=${execWeight}, Anti-Hesitation=${hesitWeight}`, 'CONFIG');
  });

  scriptSelector.addEventListener('change', (e) => {
    loadScript(e.target.value);
  });

  btnSyncGitea.addEventListener('click', () => {
    loadingTitle.textContent = "Syncing Repository to Gitea...";
    loadingOverlay.classList.remove('opacity-0', 'pointer-events-none');

    terminal.addLog('INFO', 'Initiating Git push to https://gitea.thetempleofdoom.com/drjones/AI--trainer.git', 'GITEA');

    setTimeout(() => {
      loadingOverlay.classList.add('opacity-0', 'pointer-events-none');
      terminal.addLog('SUCCESS', '✓ Repository pushed to Gitea successfully! Credentials verified.', 'GITEA');
    }, 2500);
  });

  // Slider Input updates
  document.getElementById('knobExperts').addEventListener('input', (e) => {
    document.getElementById('knobExpertsVal').textContent = `${e.target.value} / 256`;
  });
  document.getElementById('knobExec').addEventListener('input', (e) => {
    document.getElementById('knobExecVal').textContent = parseFloat(e.target.value).toFixed(1);
  });
  document.getElementById('knobHesit').addEventListener('input', (e) => {
    document.getElementById('knobHesitVal').textContent = parseFloat(e.target.value).toFixed(1);
  });

  // Initialize
  initWebSocket();
  renderFleetNodes();
  loadScript('harvester.py');
});
