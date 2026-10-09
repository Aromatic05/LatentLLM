import type { EventMessage, Media, RunRecord, StatusRecord } from './Model';

async function checked<T>(response: Response): Promise<T> {
  const payload = await response.json().catch(() => ({}));
  if (!response.ok) throw new Error(payload.detail || `HTTP ${response.status}`);
  return payload as T;
}

export async function getStatus(): Promise<StatusRecord> {
  return checked(await fetch('/api/status'));
}

export async function getRun(id: string): Promise<RunRecord> {
  return checked(await fetch(`/api/runs/${encodeURIComponent(id)}`));
}

export async function startRun(prompt: string, media: Media, expected?: string, delta = true): Promise<void> {
  await checked(await fetch('/api/runs', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ prompt, media, delta, expected: expected || null }),
  }));
}

export function watchStatus(update: (next: EventMessage) => void, connection: (online: boolean) => void): () => void {
  let socket: WebSocket | undefined;
  let retry: ReturnType<typeof setTimeout> | undefined;
  let closed = false;
  const connect = () => {
    if (closed) return;
    const scheme = location.protocol === 'https:' ? 'wss:' : 'ws:';
    socket = new WebSocket(`${scheme}//${location.host}/api/events`);
    socket.onopen = () => connection(true);
    socket.onmessage = event => {
      try {
        const message = JSON.parse(event.data) as EventMessage;
        if (message.type === 'snapshot') update(message);
      } catch { /* Ignore a single malformed telemetry event. */ }
    };
    socket.onclose = () => { connection(false); if (!closed) retry = setTimeout(connect, 1800); };
    socket.onerror = () => { socket?.close(); };
  };
  connect();
  return () => { closed = true; if (retry) clearTimeout(retry); socket?.close(); };
}
