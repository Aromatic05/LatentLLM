import { ArrowDownUp, ArrowRight, Box, Cable, Check, Clock3, LoaderCircle, PackageOpen } from 'lucide-react';
import { useWorkspace } from '../experiment/State';
import type { TransferRecord } from '../experiment/Model';

const EMPTY_TRANSFERS: TransferRecord[] = [];

function size(bytes: number) { return bytes < 1024 ? `${bytes} B` : bytes < 1024 * 1024 ? `${(bytes / 1024).toFixed(1)} KiB` : `${(bytes / 1048576).toFixed(2)} MiB`; }

export function TransferPanel() {
  const transfers = useWorkspace(s => s.status.run?.transfers ?? EMPTY_TRANSFERS);
  return <section className="panel transfers-panel" id="transfers"><div className="panel-header"><div><div className="panel-kicker"><Cable size={13} /> CROSS-NODE COMMUNICATION</div><h2>通信链路</h2></div><span className="panel-counter">{transfers.length} TRANSFERS</span></div>
    <div className="transfer-column-label"><span>PATH / PAYLOAD</span><span>SIZE</span><span>LATENCY</span></div>
    <div className="transfer-scroll">{transfers.length ? transfers.map(transfer => <div className="transfer-row" key={transfer.id}>
      <div className="transfer-path"><span className={`transfer-symbol ${transfer.media}`}><Box size={17} /></span><div><div className="transfer-route">{transfer.source}<ArrowRight size={12} />{transfer.target}</div><span className={`transfer-media ${transfer.media}`}>{transfer.media === 'kv' ? 'KV CACHE' : 'TOKEN IDS'}{transfer.tensors !== undefined ? ` · ${transfer.tensors} tensors` : ''}</span></div></div>
      <div className="transfer-size">{size(transfer.bytes)}</div><div className="transfer-latency" title={transfer.sendMs != null ? `发送端 ${transfer.sendMs.toFixed(2)} ms / 接收端 ${transfer.receiveMs?.toFixed(2)} ms` : undefined}>{transfer.durationMs != null ? <><Clock3 size={11} /> {transfer.durationMs.toFixed(1)} ms</> : <LoaderCircle size={13} className="spin" />}</div>
    </div>) : <div className="empty-transfers"><div className="empty-inline-icon"><PackageOpen size={25} /></div><strong>尚无传输记录</strong><p>选择通信模式并启动实验，Token 与 KV-Cache 的传输明细将在此显示。</p></div>}</div>
    <div className="transfer-footer"><span><ArrowDownUp size={12}/> RPC 交接耗时含传输协调，非纯网络耗时</span><span><Check size={12}/> Verified transfers only</span></div></section>;
}
