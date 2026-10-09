import { useState } from 'react';
import { Clock3, History, LoaderCircle } from 'lucide-react';
import { getRun } from './Client';
import { useWorkspace } from './State';

export function ExperimentHistory() {
  const { status, archive, setArchive } = useWorkspace();
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const history = status.history ?? [];

  async function select(id: string) {
    setError(null);
    if (!id) { setArchive(null); return; }
    setBusy(true);
    try {
      const run = await getRun(id);
      setArchive({ ...run, replayed: true });
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : '无法加载实验记录');
    } finally {
      setBusy(false);
    }
  }

  return <div className="history-picker">
    <label htmlFor="run-history"><History size={14}/> 实验历史</label>
    <select id="run-history" disabled={busy} value={archive?.id ?? ''}
      onChange={e => void select(e.target.value)}>
      <option value="">当前 / 最近一次</option>
      {history.map(item => <option value={item.id} key={item.id}>
        {item.media.toUpperCase()}{item.media === 'kv' ? (item.delta === false ? ' FULL' : ' Δ') : ''} · {item.prompt.slice(0, 17)} · {item.correct === true ? '✓' : item.correct === false ? '✗' : item.expected ? '?' : '—'}
      </option>)}
    </select>
    {busy ? <LoaderCircle className="spin" size={14}/> : <Clock3 size={13}/>}
    {error && <span className="history-error" role="alert">{error}</span>}
  </div>;
}
