/**
 * terminal.js - Rolling Cyber Terminal Console Component
 * Manages streaming logs, ANSI badge colors, auto-scrolling, filtering, and collapsing.
 */

class TerminalConsole {
  constructor() {
    this.container = document.getElementById('terminalBody');
    this.badgeCount = document.getElementById('logCountBadge');
    this.autoScrollBtn = document.getElementById('btnAutoScroll');
    this.clearBtn = document.getElementById('btnClearLogs');
    this.toggleBtn = document.getElementById('btnToggleTerminal');
    this.chevron = document.getElementById('terminalChevron');

    this.logs = [];
    this.activeFilter = 'ALL';
    this.autoScroll = true;
    this.isCollapsed = false;

    this.initListeners();
  }

  initListeners() {
    this.autoScrollBtn.addEventListener('click', () => {
      this.autoScroll = !this.autoScroll;
      this.autoScrollBtn.classList.toggle('bg-cyan-500/20', this.autoScroll);
      this.autoScrollBtn.classList.toggle('text-cyan-400', this.autoScroll);
    });

    this.clearBtn.addEventListener('click', () => {
      this.logs = [];
      this.render();
    });

    this.toggleBtn.addEventListener('click', () => {
      const termContainer = document.getElementById('terminalContainer');
      this.isCollapsed = !this.isCollapsed;
      if (this.isCollapsed) {
        termContainer.style.transform = 'translateY(calc(100% - 38px))';
        this.chevron.className = 'fa-solid fa-chevron-up';
      } else {
        termContainer.style.transform = 'translateY(0)';
        this.chevron.className = 'fa-solid fa-chevron-down';
      }
    });

    document.querySelectorAll('.log-filter-btn').forEach(btn => {
      btn.addEventListener('click', (e) => {
        document.querySelectorAll('.log-filter-btn').forEach(b => b.classList.remove('active'));
        e.target.classList.add('active');
        this.activeFilter = e.target.dataset.filter;
        this.render();
      });
    });
  }

  addLog(level, message, stage = 'SYSTEM', timestamp = null) {
    const timeStr = timestamp || new Date().toISOString().split('T')[1].slice(0, 12);
    const logItem = { level, message, stage, timestamp: timeStr };

    this.logs.push(logItem);
    if (this.logs.length > 500) this.logs.shift();

    if (this.shouldDisplay(logItem)) {
      this.appendLogElement(logItem);
    }

    this.badgeCount.textContent = `${this.logs.length} Logs`;
  }

  shouldDisplay(log) {
    if (this.activeFilter === 'ALL') return true;
    return log.level === this.activeFilter;
  }

  appendLogElement(log) {
    const div = document.createElement('div');
    div.className = 'log-line';

    let badgeClass = 'badge-info';
    if (log.level === 'HARNESS') badgeClass = 'badge-harness';
    else if (log.level === 'REWARD') badgeClass = 'badge-reward';
    else if (log.level === 'WARN') badgeClass = 'badge-warn';
    else if (log.level === 'ERROR') badgeClass = 'badge-error';

    div.innerHTML = `
      <span class="log-timestamp">[${log.timestamp}]</span>
      <span class="px-1.5 py-0.2 text-[9px] font-bold rounded uppercase ${badgeClass}">${log.level}</span>
      <span class="text-slate-500 font-bold">[${log.stage}]</span>
      <span class="text-slate-300 font-mono">${this.escapeHtml(log.message)}</span>
    `;

    this.container.appendChild(div);

    if (this.autoScroll) {
      this.container.scrollTop = this.container.scrollHeight;
    }
  }

  render() {
    this.container.innerHTML = '';
    this.logs.filter(log => this.shouldDisplay(log)).forEach(log => this.appendLogElement(log));
    this.badgeCount.textContent = `${this.logs.length} Logs`;
  }

  escapeHtml(str) {
    return str.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  }
}
