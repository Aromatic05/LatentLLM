import { useMemo } from 'react';
import { ArrowUpRight, BrainCircuit, CheckCircle2, Clock3, MessageSquareText, Sparkles } from 'lucide-react';
import { roles, useWorkspace } from '../experiment/State';

export function AgentTrace() {
  const { status, selectedAgent, setAgent } = useWorkspace();
  const run = status.run;
  const agents = run?.agents.length ? run.agents : roles;
  const selected = agents.find(agent => agent.id === selectedAgent) ?? null;
  const relevant = useMemo(() => (run?.events ?? []).filter(e => !selectedAgent || e.agent === selectedAgent || !e.agent), [run?.events, selectedAgent]);
  return (
    <section className="panel trace-panel" id="discussion">
      <div className="panel-header"><div><div className="panel-kicker"><MessageSquareText size={13} /> AGENT ACTIVITY</div><h2>智能体讨论过程</h2></div>
        <span className="panel-counter">{run?.events.length || 0} EVENTS</span>
      </div>
      <div className="agent-tabs">
        <button type="button" onClick={() => setAgent(null)} className={!selectedAgent ? 'on' : ''}>All</button>
        {agents.map(agent => <button type="button" key={agent.id} className={selectedAgent === agent.id ? 'on' : ''} onClick={() => setAgent(agent.id)}>{agent.role}</button>)}
      </div>
      <div className="trace-scroll">
        {selected && <div className="selected-agent"><div className="selected-agent-title"><BrainCircuit size={15} /> {selected.role} <span className="selected-state">{selected.status.toUpperCase()}</span></div>
          <div className="selected-agent-body">{selected.output || (selected.latentSteps ? `潜在推理：${selected.latentSteps} steps；没有生成可读讨论文本。` : '当前没有可读输出。')}</div>
          {selected.durationMs !== undefined && <div className="selected-agent-meta"><Clock3 size={12} /> 推理 {selected.durationMs.toFixed(0)} ms</div>}
        </div>}
        {relevant.length === 0 ? <div className="empty-trace"><div className="empty-symbol"><Sparkles size={22} /></div><h3>等待智能体开始协作</h3><p>启动一个真实实验后，推理步骤和跨机消息将按发生顺序显示在这里。</p></div> :
          <div className="trace-list">{relevant.map(item => <div className="trace-entry" key={item.id}><div className="trace-rail"><span className="trace-bullet" /></div><div className="trace-entry-content"><div className="trace-entry-head"><span className="trace-kind">{item.kind.replaceAll('.', ' / ')}</span><time>{new Date(item.time).toLocaleTimeString('zh-CN', { hour12: false })}</time></div><p>{item.message}</p>{item.durationMs != null && <span className="trace-duration"><Clock3 size={11} /> {item.durationMs.toFixed(1)} ms</span>}</div></div>)}</div>}
      </div>
      <div className="trace-footer"><span>{run?.media === 'kv' ? <><BrainCircuit size={13} /> Latent 模式不产生可读中间讨论</> : <><CheckCircle2 size={13} /> 仅展示真实运行事件</>}</span><ArrowUpRight size={14} /></div>
    </section>
  );
}
