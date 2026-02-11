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

const OVERLAY_MIN_INTERVAL_MS = 33;

export default function App() {
  const [lastOverlay, setLastOverlay] = useState<WsOverlay | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const lastUpdateRef = useRef(0);
  const pendingPayloadRef = useRef<WsOverlay | null>(null);
  const rafRef = useRef<number | null>(null);

  useEffect(() => {
    const scheduleFlush = () => {
      if (rafRef.current !== null) {
        return;
      }

      const flush = (ts: number) => {
        rafRef.current = null;
        const payload = pendingPayloadRef.current;
        if (!payload) {
          return;
        }

        const elapsed = ts - lastUpdateRef.current;
        if (elapsed >= OVERLAY_MIN_INTERVAL_MS) {
          lastUpdateRef.current = ts;
          pendingPayloadRef.current = null;
          setLastOverlay(payload);
          return;
        }

        scheduleFlush();
      };

      rafRef.current = window.requestAnimationFrame(flush);
    };

    const ws = new WebSocket(`${WS_BASE_URL}/ws`);
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as WsOverlay;
        pendingPayloadRef.current = payload;
        scheduleFlush();
      } catch {
        setLastOverlay(null);
      }
    };

    return () => {
      ws.close();
      if (rafRef.current !== null) {
        window.cancelAnimationFrame(rafRef.current);
        rafRef.current = null;
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
