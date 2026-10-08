import { BrainCircuit, ChevronDown } from 'lucide-react';

export function visibleOutput(output: string): string {
  const end = output.lastIndexOf('</think>');
  if (end >= 0) return output.slice(end + '</think>'.length).trim();
  if (output.trim().startsWith('<think>')) return '';
  return output.trim();
}

export function OutputView({ output }: { output: string }) {
  const visible = visibleOutput(output);
  const thoughtStart = output.indexOf('<think>');
  const thoughtEnd = output.indexOf('</think>');
  const thought = thoughtStart >= 0
    ? output.slice(thoughtStart + '<think>'.length, thoughtEnd >= 0 ? thoughtEnd : undefined).trim()
    : '';
  return <div className="agent-output">
    {visible ? <p className="agent-output-message">{visible}</p> : <p className="agent-output-warning">{thoughtStart >= 0 ? '模型只产生了思考内容，尚无可读讨论结果；可能达到生成上限。' : '当前没有可读输出。'}</p>}
    {thought && <details className="agent-thought"><summary><BrainCircuit size={13}/> 模型原始思考输出 <ChevronDown size={13}/></summary><p>{thought}</p></details>}
  </div>;
}
