import { ArrowUpRight, Clock3, Database, Network, Radio } from 'lucide-react';
import { useRun } from '../experiment/State';

function bytes(value: number) { return value >= 1048576 ? `${(value / 1048576).toFixed(2)} MiB` : value >= 1024 ? `${(value / 1024).toFixed(1)} KiB` : `${value} B`; }

export function MetricStrip() {
  const run = useRun();
  const completed = run?.transfers.filter(t => t.durationMs != null) ?? [];
  const elapsed = run?.finishedAt ? new Date(run.finishedAt).getTime() - new Date(run.startedAt).getTime() : null;
  const totalBytes = run?.transfers.reduce((sum, t) => sum + t.bytes, 0) ?? 0;
  const latency = completed.length ? completed.reduce((sum, t) => sum + (t.durationMs ?? 0), 0) / completed.length : null;
  const data = [
    { key: 'Completed transfers', value: String(completed.length), icon: Radio, unit: 'transfers', detail: '已完成的跨机交接' },
    { key: 'Payload transferred', value: completed.length ? bytes(totalBytes) : '—', icon: Database, unit: 'total bytes', detail: '已记录的应用层传输量' },
    { key: 'Avg. handoff latency', value: latency == null ? '—' : latency.toFixed(1), icon: Network, unit: 'ms', detail: '协调端 RPC 交接耗时' },
    { key: 'Experiment duration', value: elapsed == null ? '—' : (elapsed / 1000).toFixed(2), icon: Clock3, unit: 'seconds', detail: '从任务启动到结束' },
  ];
  return <div className="metric-strip">{data.map(item => { const Icon = item.icon; return <div className="metric-tile" key={item.key}><div className="metric-top"><span>{item.key}</span><Icon size={16}/></div><div className="metric-value">{item.value}<span>{item.unit}</span></div><div className="metric-detail">{item.detail}<ArrowUpRight size={12}/></div></div>; })}</div>;
}
