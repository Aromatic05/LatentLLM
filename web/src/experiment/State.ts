import { create } from 'zustand';
import type { AgentRecord, Media, RunRecord, StatusRecord } from './Model';

interface WorkspaceState {
  status: StatusRecord;
  connected: boolean;
  archive: RunRecord | null;
  setArchive: (archive: RunRecord | null) => void;
  selectedAgent: string | null;
  media: Media;
  busy: boolean;
  error: string | null;
  setStatus: (status: StatusRecord) => void;
  setConnected: (connected: boolean) => void;
  setAgent: (agent: string | null) => void;
  setMedia: (media: Media) => void;
  setBusy: (busy: boolean) => void;
  setError: (error: string | null) => void;
}

export const roles: AgentRecord[] = [
  { id: 'planner', role: 'Planner', node: 'node-a', status: 'pending' },
  { id: 'critic', role: 'Critic', node: 'node-b', status: 'pending' },
  { id: 'refiner', role: 'Refiner', node: 'node-a', status: 'pending' },
  { id: 'judger', role: 'Judger', node: 'node-b', status: 'pending' },
];

export const useWorkspace = create<WorkspaceState>((set) => ({
  status: { nodes: [], run: null, modelReady: false },
  connected: false,
  archive: null,
  setArchive: (archive) => set({ archive, selectedAgent: null }),
  selectedAgent: null,
  media: 'token',
  busy: false,
  error: null,
  setStatus: (status) => set({ status }),
  setConnected: (connected) => set({ connected }),
  setAgent: (selectedAgent) => set({ selectedAgent }),
  setMedia: (media) => set({ media }),
  setBusy: (busy) => set({ busy }),
  setError: (error) => set({ error }),
}));

/** Selected historical trace, or the real active/latest run from the coordinator. */
export const useRun = () => useWorkspace(state => state.archive ?? state.status.run);
