import { ArrowLeftRight, CheckCircle2, XCircle, CircleHelp, Clock3, Database, FlaskConical } from 'lucide-react';
import { useWorkspace } from './State';
import { visibleOutput } from '../agent/Output';
import type { Media, RunSummary } from './Model';

function bytes(size: number) {
  if (size >= 1048576) return `${(size / 1048576).toFixed(2)} MiB`;
  if (size >= 1024) return `${(size / 1024).toFixed(1)} KiB`;
  return `${size} B`;
}

function ResultCard({ media, result }: { media: Media; result?: RunSummary }) {
  const kv = media === 'kv';
  const label = !result ? 'NO EXPERIMENT' : !result.expected ? 'EXECUTED' : result.correct === true ? 'ANSWER MATCHED' : result.correct === false ? 'ANSWER MISMATCH' : 'ANSWER UNPARSED';
  return <div className={`comparison-card ${kv ? 'kv' : 'token'}`}>
    <div className="comparison-card-head"><div className="comparison-mode"><span className={`comparison-bubble ${kv ? 'kv' : 'token'}`}>
      {kv ? <Database size={17}/> : <ArrowLeftRight size={17}/>}</span><div><strong>{kv ? 'KV-Cache' : 'Token'}</strong><small>{kv ? 'LatentMAS / Continuous Memory' : 'TextMAS / Discrete Tokens'}</small></div></div>
      {result ? <span className={`comparison-status ${result.correct === false ? 'incorrect' : result.expected && result.correct == null ? 'unparsed' : !result.expected ? 'missing' : ''}`}>{result.correct === false ? <XCircle size={12}/> : result.expected && result.correct == null ? <CircleHelp size={12}/> : <CheckCircle2 size={12}/>} {label}</span> : <span className="comparison-status missing">{label}</span>}</div>
    <div className="comparison-values"><div><span>端到端总耗时</span><strong>{result ? (result.durationMs / 1000).toFixed(2) : '—'} <small>s</small></strong></div>
      <div><span>累计传输数据量</span><strong>{result ? bytes(result.bytes) : '—'}</strong></div></div>
    <div className="comparison-answer"><span>最终答案</span><p>{result?.answer ? (visibleOutput(result.answer) || '(最终输出缺少可读答案)').slice(-400) : '尚未对当前任务运行此通信模式。'}</p>{result?.expected && <div className="comparison-evaluation">参考 {result.expected} / 解析 {result.prediction || "未解析"}</div>}</div>
  </div>;
}

export function ComparePanel() {
  const { status } = useWorkspace();
  const current = status.run;
  const history = status.history ?? [];
  const matching = history.filter(run => current && run.prompt === current.prompt && run.status === 'completed');
  const last = (mode: Media) => matching.find(run => run.media === mode);
  return <section className="panel comparison-panel" id="comparison">
    <div className="panel-header"><div><div className="panel-kicker"><FlaskConical size={13}/> SAME-TASK MODE COMPARISON</div>
      <h2>Token / KV-Cache 实验对照</h2></div><span className="panel-counter">SAME PROMPT / OBSERVED DATA</span></div>
    <div className="comparison-grid"><ResultCard media="token" result={last('token')}/><ResultCard media="kv" result={last('kv')}/></div>
    <div className="comparison-footer"><Clock3 size={12}/><span>仅比较相同任务输入的最近记录。两种模式的推理算法不同，耗时差不代表纯通信收益；不显示未实测数据。</span></div>
  </section>;
}
