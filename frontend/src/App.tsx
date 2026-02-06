import { useEffect, useRef, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import StatusStrip from './components/StatusStrip';
import Nav from './components/Nav';
import LiveOps from './pages/LiveOps';
import Events from './pages/Events';
import Identities from './pages/Identities';
import Cameras from './pages/Cameras';
import Settings from './pages/Settings';
import { WS_BASE_URL } from './config';

export type WsOverlay = {
  version: string;
  type: string;
  timestamp: string;
  appearance_mode: string;
  threshold_profile: string;
  policy: {
    decision: string;
    reason_code: string;
  };
  data: {
    camera_id: number;
    faces?: Array<{
      box: [number, number, number, number];
      score: number;
      quality: string;
      label: string;
    }>;
  };
};

export default function App() {
  const [lastOverlay, setLastOverlay] = useState<WsOverlay | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const lastUpdateRef = useRef(0);
  const pendingPayloadRef = useRef<WsOverlay | null>(null);
  const pendingTimerRef = useRef<number | null>(null);

  useEffect(() => {
    const ws = new WebSocket(`${WS_BASE_URL}/ws`);
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as WsOverlay;
        const now = Date.now();
        const elapsed = now - lastUpdateRef.current;
        if (elapsed >= 100) {
          lastUpdateRef.current = now;
          setLastOverlay(payload);
          return;
        }
        pendingPayloadRef.current = payload;
        if (pendingTimerRef.current === null) {
          const delay = Math.max(100 - elapsed, 0);
          pendingTimerRef.current = window.setTimeout(() => {
            pendingTimerRef.current = null;
            if (pendingPayloadRef.current) {
              lastUpdateRef.current = Date.now();
              setLastOverlay(pendingPayloadRef.current);
              pendingPayloadRef.current = null;
            }
          }, delay);
        }
      } catch {
        setLastOverlay(null);
      }
    };
    return () => {
      ws.close();
      if (pendingTimerRef.current !== null) {
        window.clearTimeout(pendingTimerRef.current);
        pendingTimerRef.current = null;
      }
    };
  }, []);

  return (
    <div className="app-shell">
      <Nav />
      <div className="app-content">
        <StatusStrip wsConnected={wsConnected} />
        <main className="app-main">
          <Routes>
            <Route path="/" element={<LiveOps lastOverlay={lastOverlay} />} />
            <Route path="/events" element={<Events />} />
            <Route path="/identities" element={<Identities />} />
            <Route path="/cameras" element={<Cameras />} />
            <Route path="/settings" element={<Settings />} />
          </Routes>
        </main>
      </div>
    </div>
  );
}
