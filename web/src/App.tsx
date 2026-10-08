import { useEffect } from 'react';
import { Activity, ArrowUpRight, CircleDot, Clock3, Github, LayoutDashboard, Network, ShieldCheck } from 'lucide-react';
import { AgentGraph } from './agent/Graph';
import { AgentTrace } from './agent/Trace';
import { OutputView } from './agent/Output';
import { ExperimentPanel } from './experiment/Panel';
import { ComparePanel } from './experiment/Compare';
import { getStatus, watchStatus } from './experiment/Client';
import { useWorkspace } from './experiment/State';
import { TransferPanel } from './transfer/Panel';
import { LatencyChart } from './telemetry/Chart';
import { MetricStrip } from './telemetry/Metric';
import { StageChart } from './telemetry/Stages';

export default function App() {
  const { connected, status, setStatus, setConnected } = useWorkspace();
  useEffect(() => {
    getStatus().then(setStatus).catch(() => setConnected(false));
    return watchStatus(message => setStatus(message.status), setConnected);
  }, [setStatus, setConnected]);
  const run = status.run;
  const sameHost = status.nodes.length > 1 && status.nodes.every(n => n.online && n.host === status.nodes[0].host);
  return <div className="app-shell">
    <aside className="sidebar"><ExperimentPanel /></aside>
    <main className="main-area">
      <header className="topbar"><div className="breadcrumbs"><LayoutDashboard size={15}/><span>Workspace</span><span className="crumb-slash">/</span><strong>Overview</strong></div>
        <div className="topbar-actions"><span className="topbar-version">SYSTEM / V0.1</span><span className={`connection-pill ${connected ? 'connected' : ''}`}><span className="connection-led"/>{connected ? 'SYSTEM CONNECTED' : 'SYSTEM DISCONNECTED'}</span><a href="https://github.com/Gen-Verse/LatentMAS" target="_blank" rel="noreferrer" className="source-link" title="LatentMAS upstream"><Github size={17}/></a></div>
      </header>
      <div className="dashboard">
        <div className="dashboard-title"><div><div className="eyebrow"><span className="tiny-square"/> DISTRIBUTED REASONING SYSTEM <span className="eyebrow-line"/></div><h1>通信实验工作台 <span> / Overview</span></h1><p>实时观察多智能体之间的 Token 与 KV-Cache 通信、推理流程和传输性能。{sameHost && <span className="deployment-label">当前：同机双进程 / LOOPBACK</span>}</p></div>
          <div className={`run-status ${run?.status || 'idle'}`}><Activity size={17}/><div><span>EXPERIMENT STATUS</span><strong>{run ? (run.replayed ? 'ARCHIVED' : ({running:'RUNNING', completed:'COMPLETED',failed:'FAILED'}[run.status])) : 'STANDBY'}</strong></div><ArrowUpRight size={14}/></div></div>
        <MetricStrip/>
        <div className="section-bar"><span><Network size={14}/> SYSTEM OVERVIEW</span><span>REAL-TIME OBSERVABILITY <CircleDot size={12}/></span></div>
        <div className="upper-grid"><AgentGraph/><AgentTrace/></div>
        <div className="section-bar second"><span><Activity size={14}/> COMMUNICATION ANALYTICS</span><span>LAST EXPERIMENT <Clock3 size={12}/></span></div>
        <div className="lower-grid"><TransferPanel/><LatencyChart/></div>
        <div className="section-bar second"><span><Activity size={14}/> INFERENCE BREAKDOWN</span><span>SYNCHRONIZED GPU TIMINGS <Clock3 size={12}/></span></div>
        <StageChart/>
        <div className="section-bar second"><span><Activity size={14}/> EXPERIMENT COMPARISON</span><span>SAME TASK, TWO MEDIA <Clock3 size={12}/></span></div>
        <ComparePanel/>
        {run && <section className="result-panel"><div><ShieldCheck size={17} /> TASK RESULT <span>· {run.id.slice(0,8)}</span>{run.expected && <span className={`answer-grade ${run.correct === true ? 'correct' : run.correct === false ? 'incorrect' : 'unparsed'}`}>{run.correct === true ? 'ANSWER MATCHED' : run.correct === false ? 'ANSWER MISMATCH' : 'ANSWER UNPARSED'} · {run.prediction || '无提取值'} / {run.expected}</span>}</div><div className="result-content">{run.answer ? <OutputView output={run.answer}/> : (run.error || '智能体正在求解，尚未产生最终输出。')}</div></section>}
        <footer className="page-footer"><span>© LATENTLLM RESEARCH LAB <span className="footer-separator">/</span> POWERED BY LATENTMAS</span><span><span className="footer-circle"/> OBSERVED DATA ONLY <span className="footer-separator">/</span> {new Date().getFullYear()}</span></footer>
      </div>
    </main>
  </div>;
}
