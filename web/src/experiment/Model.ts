export type Media = 'token' | 'kv';
export type AgentStatus = 'pending' | 'running' | 'completed' | 'failed';
export type RunStatus = 'running' | 'completed' | 'failed';

export interface NodeRecord {
  id: string;
  host: string;
  online: boolean;
  model: string | null;
  device: string | null;
}

export interface AgentRecord {
  id: string;
  role: string;
  node: string;
  status: AgentStatus;
  output?: string;
  inputTokens?: number;
  outputTokens?: number;
  latentSteps?: number;
  durationMs?: number;
  restoreMs?: number;
  inferenceMs?: number;
  encodeMs?: number;
  cachePositions?: number;
}

export interface TransferRecord {
  id: string;
  source: string;
  target: string;
  media: Media;
  bytes: number;
  durationMs: number | null;
  status: 'sending' | 'completed' | 'failed';
  sequence?: number;
  tensors?: number;
  sendMs?: number;
  receiveMs?: number;
}

export interface EventRecord {
  id: number;
  time: string;
  kind: string;
  agent?: string;
  message: string;
  durationMs?: number;
  restoreMs?: number;
  inferenceMs?: number;
  encodeMs?: number;
  cachePositions?: number;
}

export interface RunRecord {
  id: string;
  prompt: string;
  media: Media;
  status: RunStatus;
  startedAt: string;
  finishedAt?: string;
  agents: AgentRecord[];
  transfers: TransferRecord[];
  events: EventRecord[];
  answer?: string;
  error?: string;
}

export interface StatusRecord {
  nodes: NodeRecord[];
  run: RunRecord | null;
  modelReady: boolean;
}

export interface EventMessage {
  type: 'snapshot';
  status: StatusRecord;
}
