import { useState } from 'react';
import { Activity, ArrowRight, Cpu, FlaskConical, Layers3, Play, RadioTower, Sparkles, Terminal } from 'lucide-react';
import { startRun } from './Client';
import { useWorkspace } from './State';
import type { Media } from './Model';

const sampleTasks = [
  { label: '数学推理', prompt: '有一个两位数，十位数字与个位数字之和为 11，交换数字后得到的数比原数大 27。求原来的两位数。', expected: '47' },
  { label: '逻辑验证', prompt: '三个人 A、B、C 中只有一个人说真话。A 说 B 说谎，B 说 C 说谎，C 说 A 和 B 都说谎。谁说真话？请逐步验证。', expected: 'B' },
];

export function ExperimentPanel() {
  const { status, connected, media, setMedia, busy, setBusy, error, setError } = useWorkspace();
  const [prompt, setPrompt] = useState(sampleTasks[0].prompt);
  const [expected, setExpected] = useState(sampleTasks[0].expected);
  const run = status.run;
  const active = run?.status === 'running';
  async function submit(event: React.FormEvent) {
    event.preventDefault();
    if (busy || active || !prompt.trim()) return;
    setError(null);
    setBusy(true);
    try { await startRun(prompt.trim(), media, expected.trim()); }
    catch (cause) { setError(cause instanceof Error ? cause.message : '实验启动失败'); }
    finally { setBusy(false); }
  }
  return <div className="sidebar-content">
    <div className="brand"><div className="brand-icon"><Layers3 size={21} strokeWidth={2.2} /></div><div><div className="brand-name">Latent<span>LLM</span></div><div className="brand-subtitle">DISTRIBUTED INTELLIGENCE LAB</div></div></div>
    <div className="side-divider" />
    <div className="side-caption"><FlaskConical size={14} /> EXPERIMENT CONTROL <span>01 / 02</span></div>
    <form onSubmit={submit} className="experiment-form">
      <label className="form-label" htmlFor="experiment-prompt">任务输入 <span>INPUT PROMPT</span></label>
      <textarea id="experiment-prompt" value={prompt} onChange={event => { setPrompt(event.target.value); setExpected(''); }} rows={6} spellCheck={false} placeholder="输入一个需要多个 Agent 讨论的任务…" />
      <div className="sample-tasks">{sampleTasks.map(task => <button key={task.label} type="button" onClick={() => { setPrompt(task.prompt); setExpected(task.expected); }}>{task.label}<ArrowRight size={11} /></button>)}</div>
      <label className="form-label" htmlFor="expected-answer">参考答案（可选） <span>EVALUATION TARGET</span></label>
      <input id="expected-answer" className="expected-input" value={expected} onChange={event => setExpected(event.target.value)} placeholder="仅提供时评价答案正确性" />
      <label className="form-label" id="media-label">通信方式 <span>TRANSFER MEDIA</span></label>
      <div className="mode-switch" role="group" aria-labelledby="media-label">{([{ value: 'token', title: 'Token', desc: '文本 / IDs', icon: Terminal }, { value: 'kv', title: 'KV-Cache', desc: '潜在状态', icon: Layers3 }] as const).map(option => {
        const Icon = option.icon;
        return <button type="button" key={option.value} onClick={() => setMedia(option.value as Media)} className={media === option.value ? 'mode-option selected' : 'mode-option'}><Icon size={17}/><strong>{option.title}</strong><small>{option.desc}</small></button>;
      })}</div>
      <button type="submit" className="run-button" disabled={busy || active || !prompt.trim() || !connected || !status.modelReady}><Play fill="currentColor" size={15} />{busy ? '正在提交' : active ? '实验运行中' : '启动实验'}<ArrowRight size={16} /></button>
      {!connected && <div className="form-warning">Web API 尚未连接。启动 Python 服务后即可运行。</div>}
      {connected && !status.modelReady && <div className="form-warning">推理模型尚未准备完成，请先配置本地模型与远程节点。</div>}
      {error && <div className="form-error" role="alert">{error}</div>}
    </form>
    <div className="side-divider" />
    <div className="side-caption"><RadioTower size={14} /> COMPUTE NODES <span>{status.nodes.filter(node => node.online).length} / {Math.max(2, status.nodes.length)}</span></div>
    <div className="node-list">{(status.nodes.length ? status.nodes : [{id:'node-a',host:'PC A',online:false,model:null,device:null},{id:'node-b',host:'PC B',online:false,model:null,device:null}]).map((node,index) => <div className="node-item" key={node.id}><div className={`node-indicator ${node.online ? 'online' : ''}`}><Cpu size={16} /></div><div><strong>{node.id.toUpperCase()} · {node.host || `PC ${index === 0 ? 'A' : 'B'}`}</strong><small>{node.model || 'Model not configured'}</small></div><span className={`node-light ${node.online ? 'online' : ''}`} /></div>)}</div>
    <div className="sidebar-spacer" />
    <div className="side-tip"><Sparkles size={15} /><div><strong>真实传输 · 真实指标</strong><p>不使用模拟性能数据。所有图表均由实际推理和网络事件驱动。</p></div></div>
    <div className="sidebar-bottom"><span>LATENTLLM / LAB</span><span>v0.1.0</span></div>
  </div>;
}
