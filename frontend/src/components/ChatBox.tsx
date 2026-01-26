import { useState } from 'react';

const personas = ['M.A.R.Y.A.M', 'A.D.A.M', 'R.A.I.S.Y.A'] as const;

export default function ChatBox() {
  const [persona, setPersona] = useState<(typeof personas)[number]>('R.A.I.S.Y.A');
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<string[]>([]);

  const handleSend = () => {
    if (!input.trim()) return;
    setMessages((prev) => [...prev, `${persona}: blocked by guardrails`]);
    setInput('');
  };

  return (
    <div className="deftech-card">
      <h3 className="text-lg font-semibold mb-2">RAISYA Chat</h3>
      <div className="text-xs font-mono text-white/70 mb-2">Guardrails active</div>
      <div className="h-40 overflow-y-auto bg-black/20 rounded p-2 text-sm">
        {messages.length === 0 && <div className="text-white/60">No messages yet.</div>}
        {messages.map((message, index) => (
          <div key={index} className="mb-1">
            {message}
          </div>
        ))}
      </div>
      <div className="flex gap-2 mt-2">
        <select
          className="bg-[#122454] border border-white/20 rounded px-2 py-1"
          value={persona}
          onChange={(event) => setPersona(event.target.value as (typeof personas)[number])}
        >
          {personas.map((p) => (
            <option key={p} value={p}>
              {p}
            </option>
          ))}
        </select>
        <input
          className="flex-1 bg-[#122454] border border-white/20 rounded px-2 py-1"
          value={input}
          onChange={(event) => setInput(event.target.value)}
          placeholder="Ask the assistant"
        />
        <button className="bg-red px-3 py-1 rounded" onClick={handleSend}>
          Send
        </button>
      </div>
    </div>
  );
}
