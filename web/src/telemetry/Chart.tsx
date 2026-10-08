import { BarChart3, TrendingUp } from 'lucide-react';
import ReactECharts from 'echarts-for-react';
import { useRun } from '../experiment/State';
import type { TransferRecord } from '../experiment/Model';

const EMPTY_TRANSFERS: TransferRecord[] = [];

export function LatencyChart() {
  const run = useRun();
  const transfers = run?.transfers ?? EMPTY_TRANSFERS;
  const completed = transfers.filter(item => item.durationMs != null);
  const option = {
    animationDuration: 350,
    backgroundColor: 'transparent',
    grid: { left: 50, right: 18, top: 24, bottom: 28 },
    tooltip: { trigger: 'axis', backgroundColor: '#111d2c', borderColor: '#2b3e54', textStyle: { color: '#e6edf6' } },
    xAxis: { type: 'category', data: completed.map((t, i) => `#${i + 1}`), axisLine: { lineStyle: { color: '#334155' } }, axisLabel: { color: '#94a3b8', fontSize: 11 }, axisTick: { show: false } },
    yAxis: { type: 'value', name: 'ms', nameTextStyle: { color: '#64748b' }, splitLine: { lineStyle: { color: '#202d3b' } }, axisLabel: { color: '#64748b', fontSize: 11 } },
    series: [{ type: 'bar', barWidth: 26, data: completed.map(t => ({ value: t.durationMs, itemStyle: { color: t.media === 'kv' ? '#ab8eff' : '#48d7be', borderRadius: [4, 4, 0, 0] } })) }],
  };
  return <section className="panel chart-panel" id="performance"><div className="panel-header"><div><div className="panel-kicker"><TrendingUp size={13} /> TELEMETRY / OBSERVABILITY</div><h2>交接调用时延</h2></div><span className="panel-counter">MS PER TRANSFER</span></div>
    {completed.length ? <div className="chart-content"><ReactECharts option={option} style={{ height: '100%', width: '100%' }} notMerge /></div> : <div className="empty-chart"><BarChart3 size={31} strokeWidth={1.4} /><strong>暂无测量数据</strong><p>运行真实通信实验后生成时延图表</p></div>}
    <div className="chart-footer"><span><i className="key-dot mint"/> Token</span><span><i className="key-dot violet"/> KV-Cache</span><span className="chart-unit">COORDINATOR RPC · LOOPBACK WHEN LOCAL</span></div></section>;
}
