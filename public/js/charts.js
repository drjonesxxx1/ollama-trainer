/**
 * charts.js - Chart.js Telemetry Engine
 * Renders GRPO Reward Convergence and MoE Expert Retention Heatmap
 */

class TelemetryCharts {
  constructor() {
    this.grpoChart = null;
    this.moeChart = null;
    this.initGRPOChart();
    this.initMoEChart();
  }

  initGRPOChart() {
    const ctx = document.getElementById('grpoChart').getContext('2d');
    
    // Gradient fills
    const execGrad = ctx.createLinearGradient(0, 0, 0, 200);
    execGrad.addColorStop(0, 'rgba(0, 255, 157, 0.4)');
    execGrad.addColorStop(1, 'rgba(0, 255, 157, 0.0)');

    const hesitGrad = ctx.createLinearGradient(0, 0, 0, 200);
    hesitGrad.addColorStop(0, 'rgba(255, 184, 0, 0.4)');
    hesitGrad.addColorStop(1, 'rgba(255, 184, 0, 0.0)');

    this.grpoChart = new Chart(ctx, {
      type: 'line',
      data: {
        labels: Array.from({ length: 20 }, (_, i) => i * 15),
        datasets: [
          {
            label: 'Hard Execution Reward (R_exec)',
            data: [0.2, 0.8, 1.4, 1.9, 2.2, 2.5, 2.7, 2.8, 2.85, 2.9, 2.92, 2.95, 2.98, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            borderColor: '#00FF9D',
            backgroundColor: execGrad,
            borderWidth: 2,
            tension: 0.4,
            fill: true
          },
          {
            label: 'Anti-Hesitation Penalty (R_anti_hesit)',
            data: [-1.8, -1.2, -0.5, 0.2, 0.8, 1.2, 1.5, 1.7, 1.85, 1.9, 1.95, 1.98, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0],
            borderColor: '#FFB800',
            backgroundColor: hesitGrad,
            borderWidth: 2,
            tension: 0.4,
            fill: true
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#94A3B8', font: { family: 'Fira Code', size: 10 } }
          }
        },
        scales: {
          x: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Fira Code', size: 10 } }
          },
          y: {
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Fira Code', size: 10 } }
          }
        }
      }
    });
  }

  initMoEChart() {
    const ctx = document.getElementById('moeChart').getContext('2d');
    
    // Sample layer depth blocks (Layers 1 through 61)
    const layers = ['L1-10', 'L11-20', 'L21-30', 'L31-40', 'L41-50', 'L51-61'];

    this.moeChart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels: layers,
        datasets: [
          {
            label: 'Retained Experts (Domain / CLI)',
            data: [64, 64, 64, 64, 64, 64],
            backgroundColor: '#00F0FF',
            borderRadius: 4
          },
          {
            label: 'Pruned Experts (Trivia / Botany)',
            data: [192, 192, 192, 192, 192, 192],
            backgroundColor: 'rgba(255, 51, 102, 0.25)',
            borderColor: 'rgba(255, 51, 102, 0.5)',
            borderWidth: 1,
            borderRadius: 4
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { color: '#94A3B8', font: { family: 'Fira Code', size: 10 } }
          }
        },
        scales: {
          x: {
            stacked: true,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Fira Code', size: 10 } }
          },
          y: {
            stacked: true,
            grid: { color: 'rgba(255, 255, 255, 0.05)' },
            ticks: { color: '#64748B', font: { family: 'Fira Code', size: 10 } }
          }
        }
      }
    });
  }

  updateGRPO(step, reward) {
    if (this.grpoChart) {
      this.grpoChart.data.labels.push(step);
      if (this.grpoChart.data.labels.length > 25) this.grpoChart.data.labels.shift();
      this.grpoChart.data.datasets[0].data.push(reward);
      if (this.grpoChart.data.datasets[0].data.length > 25) this.grpoChart.data.datasets[0].data.shift();
      this.grpoChart.update();
    }
  }

  updateMoE(retained) {
    if (this.moeChart) {
      const dropped = 256 - retained;
      this.moeChart.data.datasets[0].data = Array(6).fill(retained);
      this.moeChart.data.datasets[1].data = Array(6).fill(dropped);
      this.moeChart.update();
    }
  }
}
