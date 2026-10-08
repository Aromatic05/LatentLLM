import { useMemo } from 'react';
import { ReactFlow, Background, BackgroundVariant, MarkerType, type Edge } from '@xyflow/react';
import { Activity, Expand, Network } from 'lucide-react';
import { AgentCard, type AgentNode } from './Node';
import { roles, useWorkspace } from '../experiment/State';

const nodeTypes = { agent: AgentCard };
const positions = [
  { x: 12, y: 88 }, { x: 260, y: 88 }, { x: 508, y: 88 }, { x: 756, y: 88 },
];

export function AgentGraph() {
  const { status, connected, selectedAgent, setAgent } = useWorkspace();
  const agents = status.run?.agents.length ? status.run.agents : roles;
  const transfers = status.run?.transfers ?? [];
  const nodes = useMemo<AgentNode[]>(() => agents.map((agent, index) => ({
    id: agent.id, type: 'agent', position: positions[index] || { x: 12 + 248 * index, y: 88 },
    data: { agent, connected, selected: selectedAgent === agent.id },
    draggable: false, selectable: true,
  })), [agents, connected, selectedAgent]);
  const edges = useMemo<Edge[]>(() => agents.slice(1).map((agent, index) => {
    const prev = agents[index];
    const transfer = transfers.find(entry => entry.source === prev.id && entry.target === agent.id);
    const isKv = transfer?.media === 'kv';
    const color = transfer ? (isKv ? '#ab8eff' : '#48d7be') : '#344358';
    return {
      id: `${prev.id}-${agent.id}`, source: prev.id, target: agent.id,
      animated: transfer?.status === 'sending',
      label: transfer ? (isKv ? 'KV-CACHE' : 'TOKEN') : undefined,
      style: { stroke: color, strokeWidth: transfer ? 2.5 : 1.5, strokeDasharray: transfer ? undefined : '5 5' },
      labelStyle: { fill: color, fontSize: 9, fontWeight: 700, letterSpacing: 1.4 },
      labelBgStyle: { fill: '#0c1420', fillOpacity: 0.95 },
      markerEnd: { type: MarkerType.ArrowClosed, color },
    };
  }), [agents, transfers]);

  return (
    <section className="panel graph-panel" id="topology">
      <div className="panel-header">
        <div><div className="panel-kicker"><Network size={13} /> DISTRIBUTED TOPOLOGY</div><h2>Agent 网络拓扑</h2></div>
        <div className="panel-header-extra"><span className="micro-label"><Activity size={12} /> {status.run ? 'LIVE WORKFLOW' : 'PLANNED WORKFLOW'}</span><Expand size={15} /></div>
      </div>
      <div className="topology-meta">
        <span><i className="key-dot mint" /> Token Channel</span>
        <span><i className="key-dot violet" /> KV-Cache Channel</span>
        <span><i className="key-dot gray" /> No Transfer</span>
      </div>
      <div className="graph-canvas">
        <ReactFlow nodes={nodes} edges={edges} nodeTypes={nodeTypes} fitView fitViewOptions={{ padding: 0.15 }}
          nodesDraggable={false} nodesConnectable={false} elementsSelectable={true}
          onNodeClick={(_, node) => setAgent(node.id)} proOptions={{ hideAttribution: true }} minZoom={0.3} maxZoom={1.3}>
          <Background color="#27374a" gap={20} size={1} variant={BackgroundVariant.Dots} />
        </ReactFlow>
      </div>
      <div className="graph-footer"><span><span className="ping" /> {status.run ? '同步运行事件' : '等待启动实验'}</span><span>点击节点检查 Agent 状态 <span className="arrow">↗</span></span></div>
    </section>
  );
}
