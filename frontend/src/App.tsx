import { useEffect, useState } from 'react';
import { Route, Routes } from 'react-router-dom';
import StatusStrip from './components/StatusStrip';
import Nav from './components/Nav';
import LiveOps from './pages/LiveOps';
import Events from './pages/Events';
import Identities from './pages/Identities';
import Cameras from './pages/Cameras';
import Settings from './pages/Settings';

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

  useEffect(() => {
    const ws = new WebSocket('ws://localhost:8000/ws');
    ws.onopen = () => setWsConnected(true);
    ws.onclose = () => setWsConnected(false);
    ws.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data) as WsOverlay;
        setLastOverlay(payload);
      } catch {
        setLastOverlay(null);
      }
    };
    return () => ws.close();
  }, []);

  return (
    <div className="min-h-screen flex">
      <Nav />
      <div className="flex-1 flex flex-col">
        <StatusStrip wsConnected={wsConnected} />
        <main className="p-6 space-y-6">
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
