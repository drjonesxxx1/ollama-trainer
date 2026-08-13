/**
 * pipeline.js - Interactive Canvas Pipeline Visualizer
 * Renders nodes, dynamic connections, glowing particle pulses, and interactive states.
 */

class PipelineVisualizer {
  constructor(canvasId) {
    this.canvas = document.getElementById(canvasId);
    this.ctx = this.canvas.getContext('2d');
    this.nodes = [
      { id: 'harvester', name: '1. Config Harvester', icon: '⚡', x: 0.12, y: 0.5, status: 'idle', desc: 'Scrapes /etc/dnsmasq.d & ADB nodes' },
      { id: 'pruner', name: '2. MoE Expert Pruner', icon: '✂️', x: 0.28, y: 0.5, status: 'idle', desc: 'Drops 70% dormant trivia experts (256->64)' },
      { id: 'harness', name: '3. Execution Gym', icon: '🛡️', x: 0.44, y: 0.5, status: 'idle', desc: 'Command safety sandbox & STDERR check' },
      { id: 'grpo', name: '4. Unsloth GRPO', icon: '🧠', x: 0.60, y: 0.5, status: 'idle', desc: 'Hard execution & anti-hesitation rewards' },
      { id: 'export', name: '5. GGUF Quantizer', icon: '📦', x: 0.76, y: 0.5, status: 'idle', desc: 'Dual-offload Q4_K_M GGUF packaging' },
      { id: 'telemetry', name: '6. State Eye Daemon', icon: '👁️', x: 0.90, y: 0.5, status: 'idle', desc: '10.30.20.1 live state context injector' }
    ];
    this.particles = [];
    this.activeStage = null;
    this.animId = null;

    this.resize();
    window.addEventListener('resize', () => this.resize());
    this.initParticles();
    this.animate();
  }

  resize() {
    const parent = this.canvas.parentElement;
    this.canvas.width = parent.clientWidth;
    this.canvas.height = parent.clientHeight;
  }

  initParticles() {
    this.particles = [];
    for (let i = 0; i < 20; i++) {
      this.particles.push({
        progress: Math.random(),
        speed: 0.003 + Math.random() * 0.004,
        size: 3 + Math.random() * 2
      });
    }
  }

  setActiveStage(stageId) {
    this.activeStage = stageId;
    this.nodes.forEach(node => {
      if (node.id === stageId) {
        node.status = 'running';
      } else if (this.getStageIndex(node.id) < this.getStageIndex(stageId)) {
        node.status = 'completed';
      } else {
        node.status = 'idle';
      }
    });
  }

  setCompleted() {
    this.nodes.forEach(n => n.status = 'completed');
    this.activeStage = null;
  }

  reset() {
    this.nodes.forEach(n => n.status = 'idle');
    this.activeStage = null;
  }

  getStageIndex(id) {
    return this.nodes.findIndex(n => n.id === id);
  }

  draw() {
    const w = this.canvas.width;
    const h = this.canvas.height;
    this.ctx.clearRect(0, 0, w, h);

    // 1. Draw Connecting Lines
    for (let i = 0; i < this.nodes.length - 1; i++) {
      const n1 = this.nodes[i];
      const n2 = this.nodes[i + 1];
      const x1 = n1.x * w;
      const y1 = n1.y * h;
      const x2 = n2.x * w;
      const y2 = n2.y * h;

      const isPassed = this.activeStage && this.getStageIndex(n1.id) <= this.getStageIndex(this.activeStage);

      this.ctx.beginPath();
      this.ctx.moveTo(x1, y1);
      this.ctx.lineTo(x2, y2);
      this.ctx.strokeStyle = isPassed ? 'rgba(0, 240, 255, 0.6)' : 'rgba(51, 65, 85, 0.5)';
      this.ctx.lineWidth = isPassed ? 3 : 2;
      this.ctx.setLineDash(isPassed ? [] : [5, 5]);
      this.ctx.stroke();
      this.ctx.setLineDash([]);
    }

    // 2. Draw Moving Data Particles
    if (this.activeStage) {
      this.particles.forEach(p => {
        p.progress += p.speed;
        if (p.progress > 1) p.progress = 0;

        const totalSegments = this.nodes.length - 1;
        const currentSegment = Math.floor(p.progress * totalSegments);
        const segProgress = (p.progress * totalSegments) - currentSegment;

        const n1 = this.nodes[currentSegment];
        const n2 = this.nodes[currentSegment + 1];

        if (n1 && n2) {
          const px = (n1.x + (n2.x - n1.x) * segProgress) * w;
          const py = (n1.y + (n2.y - n1.y) * segProgress) * h;

          this.ctx.beginPath();
          this.ctx.arc(px, py, p.size, 0, Math.PI * 2);
          this.ctx.fillStyle = '#00F0FF';
          this.ctx.shadowColor = '#00F0FF';
          this.ctx.shadowBlur = 10;
          this.ctx.fill();
          this.ctx.shadowBlur = 0;
        }
      });
    }

    // 3. Draw Nodes
    this.nodes.forEach(node => {
      const nx = node.x * w;
      const ny = node.y * h;

      // Glow effect for active/completed nodes
      if (node.status === 'running') {
        this.ctx.beginPath();
        this.ctx.arc(nx, ny, 32, 0, Math.PI * 2);
        this.ctx.fillStyle = 'rgba(0, 240, 255, 0.15)';
        this.ctx.shadowColor = '#00F0FF';
        this.ctx.shadowBlur = 20;
        this.ctx.fill();
        this.ctx.shadowBlur = 0;
      }

      // Outer Circle
      this.ctx.beginPath();
      this.ctx.arc(nx, ny, 24, 0, Math.PI * 2);
      if (node.status === 'running') {
        this.ctx.fillStyle = '#0F172A';
        this.ctx.strokeStyle = '#00F0FF';
        this.ctx.lineWidth = 3;
      } else if (node.status === 'completed') {
        this.ctx.fillStyle = 'rgba(0, 255, 157, 0.2)';
        this.ctx.strokeStyle = '#00FF9D';
        this.ctx.lineWidth = 2;
      } else {
        this.ctx.fillStyle = '#090D16';
        this.ctx.strokeStyle = '#334155';
        this.ctx.lineWidth = 2;
      }
      this.ctx.fill();
      this.ctx.stroke();

      // Icon
      this.ctx.font = '16px sans-serif';
      this.ctx.textAlign = 'center';
      this.ctx.textBaseline = 'middle';
      this.ctx.fillText(node.icon, nx, ny);

      // Node Name Label
      this.ctx.font = 'bold 11px Outfit, sans-serif';
      this.ctx.fillStyle = node.status === 'running' ? '#00F0FF' : node.status === 'completed' ? '#00FF9D' : '#94A3B8';
      this.ctx.fillText(node.name, nx, ny + 42);

      // Status Badge
      this.ctx.font = '9px Fira Code, monospace';
      this.ctx.fillStyle = '#64748B';
      this.ctx.fillText(node.status.toUpperCase(), nx, ny + 56);
    });
  }

  animate() {
    this.draw();
    this.animId = requestAnimationFrame(() => this.animate());
  }
}
