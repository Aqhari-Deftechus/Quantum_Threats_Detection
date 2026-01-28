import { useEffect, useMemo, useState } from 'react';

type ChatContext = {
  id: string;
  label: string;
  type: 'alert' | 'camera' | 'track' | 'event';
  severity?: string;
};

type AuditEvent = {
  id: string;
  timestamp: string;
  category: 'guardrail_block';
  redacted_text: string;
  reason: string;
};

const blockedPatterns = [
  /(^|\b)(hi|hello|hey|hola|sup|yo)\b/i,
  /(are\s+you\s+married|married|single|boyfriend|girlfriend|dating)/i,
  /(dah\s+kahwin|dah\s+kawin|kahwin|kawin|tunang|jodoh)/i,
  /(joke|lawak|meme|kelakar)/i,
  /(age|umur|birthday|born|religion|agama|politics|seks|sex)/i
];

const allowlistIntents = [
  { name: 'system_status', keywords: ['status', 'uptime', 'latency', 'health', 'system'] },
  { name: 'alert_explain', keywords: ['why', 'reason', 'explain', 'alert', 'threat'] },
  { name: 'camera_location', keywords: ['camera', 'location', 'where', 'zone', 'sector'] },
  { name: 'poi', keywords: ['poi', 'person of interest', 'watchlist', 'suspect', 'identity'] },
  { name: 'evidence', keywords: ['evidence', 'clip', 'recording', 'footage'] }
];

const blockedReply = 'This interface is restricted to mission-related questions only.';

function redactText(input: string) {
  if (!input.trim()) {
    return '';
  }
  return '[REDACTED]';
}

function matchesBlocked(input: string) {
  return blockedPatterns.some((pattern) => pattern.test(input));
}

function resolveIntent(input: string) {
  const lowered = input.toLowerCase();
  return allowlistIntents.find((intent) =>
    intent.keywords.some((keyword) => lowered.includes(keyword))
  );
}

export default function ChatBox({ context, autoOpen }: { context: ChatContext | null; autoOpen?: boolean }) {
  const [isOpen, setIsOpen] = useState(false);
  const [input, setInput] = useState('');
  const [messages, setMessages] = useState<string[]>([]);
  const [auditLog, setAuditLog] = useState<AuditEvent[]>([]);

  useEffect(() => {
    if (autoOpen) {
      setIsOpen(true);
    }
  }, [autoOpen]);

  const activeContextLabel = useMemo(() => {
    if (!context) return null;
    return `${context.type.toUpperCase()} ${context.label}`;
  }, [context]);

  const guardrailCheck = (value: string) => {
    const trimmed = value.trim();
    if (!trimmed) return { allowed: false, reason: 'empty' };
    if (matchesBlocked(trimmed)) return { allowed: false, reason: 'blocked_pattern' };

    const intent = resolveIntent(trimmed);
    if (!intent) return { allowed: false, reason: 'unknown_intent' };
    if (intent.name === 'system_status') return { allowed: true, reason: 'system_status' };

    if (!context) return { allowed: false, reason: 'missing_context' };

    const lowered = trimmed.toLowerCase();
    const requiresContext = [context.id, context.label].some((value) =>
      lowered.includes(value.toLowerCase())
    );
    if (!requiresContext) {
      return { allowed: false, reason: 'context_not_referenced' };
    }

    return { allowed: true, reason: intent.name };
  };

  const handleSend = () => {
    const outcome = guardrailCheck(input);
    if (!outcome.allowed) {
      const auditEvent: AuditEvent = {
        id: crypto.randomUUID(),
        timestamp: new Date().toISOString(),
        category: 'guardrail_block',
        redacted_text: redactText(input),
        reason: outcome.reason
      };
      setAuditLog((prev) => [auditEvent, ...prev].slice(0, 25));
      console.warn('Guardrail block', auditEvent);
      setMessages((prev) => [...prev, `R.A.I.S.Y.A: ${blockedReply}`]);
      setInput('');
      return;
    }

    setMessages((prev) => [
      ...prev,
      `R.A.I.S.Y.A: Acknowledged. Provide confirmation on ${activeContextLabel ?? 'system status'}.`
    ]);
    setInput('');
  };

  const toggleOpen = () => setIsOpen((prev) => !prev);

  return (
    <div className={`chatbox ${isOpen ? 'chatbox-open' : ''}`}>
      <button className="chatbox-toggle" onClick={toggleOpen}>
        R.A.I.S.Y.A
        <span className="chatbox-toggle-sub">Operator Assistant</span>
      </button>
      <div className="chatbox-panel">
        <div className="chatbox-header">
          <div>
            <div className="chatbox-title">Mission Assistant</div>
            <div className="chatbox-subtitle">Guardrails enforced • Mission scope only</div>
          </div>
          {activeContextLabel && (
            <div className="chatbox-context">Context: {activeContextLabel}</div>
          )}
        </div>
        <div className="chatbox-body">
          {messages.length === 0 && (
            <div className="chatbox-empty">Awaiting operator query.</div>
          )}
          {messages.map((message, index) => (
            <div key={index} className="chatbox-message">
              {message}
            </div>
          ))}
        </div>
        <div className="chatbox-input-row">
          <input
            className="chatbox-input"
            value={input}
            onChange={(event) => setInput(event.target.value)}
            placeholder="Ask about alert, camera, POI, evidence, or system status"
          />
          <button className="chatbox-send" onClick={handleSend}>
            Send
          </button>
        </div>
        {auditLog.length > 0 && (
          <div className="chatbox-audit">
            <div className="chatbox-audit-title">Audit (last {auditLog.length})</div>
            <div className="chatbox-audit-list">
              {auditLog.map((event) => (
                <div key={event.id} className="chatbox-audit-item">
                  {event.timestamp} • {event.reason}
                </div>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
