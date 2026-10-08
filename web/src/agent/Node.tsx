import { Handle, Position, type Node, type NodeProps } from '@xyflow/react';
import { BrainCircuit, CircleCheck, LoaderCircle, Radio, Server } from 'lucide-react';
import type { AgentRecord } from '../experiment/Model';

export type AgentNode = Node<{ agent: AgentRecord; connected: boolean; selected: boolean }, 'agent'>;

export function AgentCard({ data }: NodeProps<AgentNode>) {
  const { agent, connected, selected } = data;
  const active = agent.status === 'running';
  return (
    <div className={`agent-node ${selected ? 'selected' : ''} ${active ? 'active' : ''}`}>
      <Handle type="target" position={Position.Left} className="agent-handle" />
      <div className="agent-node-top"><span className="node-avatar"><BrainCircuit size={18} /></span>
        <span className={`status-dot ${agent.status}`} />
      </div>
      <div className="agent-node-name">{agent.role}</div>
      <div className="agent-node-host"><Server size={11} /> {agent.node === 'node-a' ? 'WORKSTATION A' : 'WORKSTATION B'}</div>
      <div className="agent-node-bottom">
        {active ? <><LoaderCircle size={11} className="spin" /> RUNNING</> : agent.status === 'completed' ? <><CircleCheck size={11} /> COMPLETE</> : connected ? <><Radio size={11} /> STANDBY</> : <><Radio size={11} /> WAITING</>}
      </div>
      <Handle type="source" position={Position.Right} className="agent-handle" />
    </div>
  );
}
