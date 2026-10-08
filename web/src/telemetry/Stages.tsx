import ReactECharts from 'echarts-for-react';
import { Activity, Layers3 } from 'lucide-react';
import { useWorkspace } from '../experiment/State';
import type { AgentRecord } from '../experiment/Model';

const EMPTY_AGENTS: AgentRecord[] = [];

export function StageChart() {
  const agents = useWorkspace(state => state.status.run?.agents ?? EMPTY_AGENTS);
  const measured = agents.filter(agent => agent.durationMs !== undefined);
  const stages = [
    { label: '恢复 · Restore', name: 'restoreMs', color: '#56a9ed' },
    { label: '模型 · Inference', name: 'inferenceMs', color: '#49d7bb' },
    { label: '编码 · Encode', name: 'encodeMs', color: '#ad94fa' },
  ] as const;
  const option = {
    animationDuration: 350,
    grid: { left: 92, right: 42, top: 28, bottom: 24 },
    tooltip: {
      trigger: 'axis', axisPointer: { type: 'shadow' },
      backgroundColor: '#111d2c', borderColor: '#2b3e54', textStyle: { color: '#e6edf6' },
      valueFormatter: (v: number) => `${Number(v).toFixed(2)} ms`,
    },
    xAxis: { type: 'value', name: 'ms', nameTextStyle: { color: '#71859d' },
      axisLine: { show: false }, axisLabel: { color: '#71859d' }, splitLine: { lineStyle: { color: '#263449' } } },
    yAxis: { type: 'category', data: measured.map(agent => agent.role),
      axisLabel: { color: '#b6c9df', fontSize: 11 }, axisLine: { show: false }, axisTick: { show: false } },
    series: stages.map(stage => ({
      name: stage.label, type: 'bar', stack: 'elapsed', barWidth: 18,
      itemStyle: { color: stage.color },
      data: measured.map(agent => Number(agent[stage.name] || 0)),
    })),
  };
  return <section className="panel stages-panel" id="stages">
    <div className="panel-header"><div><div className="panel-kicker"><Layers3 size={13} /> MODEL EXECUTION PROFILER</div><h2>推理阶段耗时</h2></div>
      <span className="panel-counter">RESTORE / INFERENCE / ENCODE</span></div>
    {measured.length ? <div className="stage-body"><ReactECharts option={option} style={{ width: '100%', height: '100%' }} notMerge /></div> :
      <div className="empty-stage"><Activity size={25}/><strong>尚无推理分段记录</strong><p>运行实验后按 Agent 展示缓存恢复、模型推理和载荷编码耗时。</p></div>}
    <div className="stage-footer"><span className="stage-key"><i className="key-dot stage-restore"/> 恢复（含 CPU→GPU）</span>
      <span className="stage-key"><i className="key-dot mint"/> 模型推理</span>
      <span className="stage-key"><i className="key-dot violet"/> 编码（含 GPU→CPU）</span>
      <span className="stage-note">同步 CUDA 测量，非纯 GPU Kernel 时间</span></div>
  </section>;
}
