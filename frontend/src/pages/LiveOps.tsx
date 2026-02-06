import { useEffect, useMemo, useRef, useState } from 'react';
import { WsOverlay } from '../App';
import WebRTCPlayer from '../components/WebRTCPlayer';
import { fetchCameraHealth, fetchCameras, fetchStatus, Camera, CameraHealth, StatusResponse } from '../api';

type LiveOpsMode = 'stealth' | 'attraction';

type ThreatEvent = {
  id: string;
  label: string;
  time: string;
  severity: 'info' | 'warn' | 'high';
  detail: string;
  isMatch?: boolean;
};

type WeaponDetection = {
  id: string;
  box: [number, number, number, number];
  confidence: number;
};

const emptyStatus: StatusResponse = {
  system: 'UNKNOWN',
  inference: 'CPU',
  matcher: 'unknown',
  matcher_index_status: 'unknown',
  evidence: 'UNKNOWN',
  cameras_live: 0,
  cameras_total: 0,
  cameras_down: 0,
  cameras_avg_fps: 0,
  cameras_avg_latency_ms: 0,
  cameras_queue_depth: 0,
  cameras_dropped_frames: 0,
  ws_status: 'UNKNOWN',
  timestamp: new Date().toISOString()
};

const defaultFrameDims = { width: 640, height: 480 };

export default function LiveOps({ lastOverlay }: { lastOverlay: WsOverlay | null }) {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [cameraHealth, setCameraHealth] = useState<Record<number, CameraHealth>>({});
  const [mode, setMode] = useState<LiveOpsMode>('stealth');
  const [frameDims, setFrameDims] = useState(defaultFrameDims);
  const [viewportDims, setViewportDims] = useState({ width: 0, height: 0 });
  const [matchPulse, setMatchPulse] = useState(false);
  const [refPulse, setRefPulse] = useState(false);
  const [heroFocus, setHeroFocus] = useState(false);
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const matchTimer = useRef<number | null>(null);

  useEffect(() => {
    let active = true;
    const load = async () => {
      try {
        const data = await fetchCameras();
        if (active) {
          setCameras(data);
        }
      } catch {
        if (active) {
          setCameras([]);
        }
      }
    };
    load();
    const interval = setInterval(load, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let active = true;
    const loadStatus = async () => {
      try {
        const data = await fetchStatus();
        if (active) {
          setStatus(data);
        }
      } catch {
        if (active) {
          setStatus(emptyStatus);
        }
      }
    };
    loadStatus();
    const interval = setInterval(loadStatus, 2000);
    return () => {
      active = false;
      clearInterval(interval);
    };
  }, []);

  useEffect(() => {
    let active = true;
    const loadHealth = async () => {
      try {
        const results = await Promise.all(
          cameras.slice(0, 1).map(async (camera) => [camera.id, await fetchCameraHealth(camera.id)] as const)
        );
        if (active) {
          const next: Record<number, CameraHealth> = {};
          results.forEach(([id, health]) => {
            next[id] = health;
          });
          setCameraHealth(next);
        }
      } catch {
        if (active) {
          setCameraHealth({});
        }
      }
    };
    if (cameras.length) {
      loadHealth();
      const interval = setInterval(loadHealth, 4000);
      return () => {
        active = false;
        clearInterval(interval);
      };
    }
    return undefined;
  }, [cameras]);

  useEffect(() => {
    if (!viewportRef.current) return undefined;
    const observer = new ResizeObserver((entries) => {
      const entry = entries[0];
      if (entry) {
        const { width, height } = entry.contentRect;
        setViewportDims({ width, height });
      }
    });
    observer.observe(viewportRef.current);
    return () => observer.disconnect();
  }, []);

  useEffect(() => {
    document.body.classList.toggle('hero-focus', heroFocus);
    return () => document.body.classList.remove('hero-focus');
  }, [heroFocus]);

  const heroCamera = cameras[0];
  const health = heroCamera ? cameraHealth[heroCamera.id] : undefined;
  const overlayFaces = lastOverlay?.data?.faces ?? [];
  const heroOverlayFaces = heroCamera && lastOverlay?.data?.camera_id === heroCamera.id ? overlayFaces : [];
  const visibleFaces = heroOverlayFaces.slice(0, 3);

  const watchlistFaces = heroOverlayFaces.filter((face) => face.label.toLowerCase().includes('watchlist'));
  const primaryFace = heroOverlayFaces[0];
  const hasMatch = watchlistFaces.length > 0;

  useEffect(() => {
    if (!hasMatch) return undefined;
    setMatchPulse(true);
    setRefPulse(true);
    if (matchTimer.current) {
      window.clearTimeout(matchTimer.current);
    }
    matchTimer.current = window.setTimeout(() => {
      setMatchPulse(false);
      setRefPulse(false);
    }, 1600);
    return () => {
      if (matchTimer.current) {
        window.clearTimeout(matchTimer.current);
      }
    };
  }, [hasMatch]);

  const threatLabel = watchlistFaces.length ? 'HIGH THREAT' : 'MONITORING';
  const threatTone = watchlistFaces.length ? 'high' : 'normal';

  const fusionScore = watchlistFaces.length ? 92 : 74;
  const agentInsight = watchlistFaces.length
    ? 'Agent Insight: Subject matches watchlist profile. Escalate to supervisor.'
    : 'Agent Insight: Corridor flow normal. No hostile indicators detected.';

  const hudMeta = {
    crowdCount: heroOverlayFaces.length,
    loiterTimer: watchlistFaces.length ? '00:12' : '—',
    restrictedZone: false,
    weaponDetected: false
  };

  const timelineEvents: ThreatEvent[] = useMemo(() => {
    const events: ThreatEvent[] = [];
    if (watchlistFaces.length) {
      events.push({
        id: 'watchlist-hit',
        label: 'Watchlist Match',
        time: new Date().toLocaleTimeString(),
        severity: 'high',
        detail: `${watchlistFaces[0]?.label ?? 'Unknown'} matched (${watchlistFaces.length} face${watchlistFaces.length > 1 ? 's' : ''})`,
        isMatch: true
      });
    }
    if (!events.length && heroOverlayFaces.length) {
      events.push({
        id: 'face-detected',
        label: 'Face Detected',
        time: new Date().toLocaleTimeString(),
        severity: 'warn',
        detail: 'Identity pending confirmation'
      });
    }
    if (!events.length) {
      events.push({
        id: 'patrol',
        label: 'Patrol Sweep',
        time: new Date().toLocaleTimeString(),
        severity: 'info',
        detail: 'No anomalies detected'
      });
    }
    events.push({
      id: 'restricted-zone',
      label: 'Restricted Area',
      time: new Date().toLocaleTimeString(),
      severity: 'info',
      detail: 'Zone overlay armed'
    });
    return events;
  }, [heroOverlayFaces, watchlistFaces]);

  const overlayTransform = useMemo(() => {
    if (!viewportDims.width || !viewportDims.height) {
      return { scale: 1, offsetX: 0, offsetY: 0 };
    }
    const scale = Math.max(viewportDims.width / frameDims.width, viewportDims.height / frameDims.height);
    const renderWidth = frameDims.width * scale;
    const renderHeight = frameDims.height * scale;
    const offsetX = (viewportDims.width - renderWidth) / 2;
    const offsetY = (viewportDims.height - renderHeight) / 2;
    return { scale, offsetX, offsetY };
  }, [frameDims, viewportDims]);

  const heroName = heroCamera?.name ?? 'Cam01 Corridor';
  const watchlistLabel = watchlistFaces[0]?.label ?? '';
  const referenceImage = watchlistLabel ? `/static/watchlist/${watchlistLabel}.jpg` : '';

  const weaponDetections: WeaponDetection[] = useMemo(() => {
    const incoming = (lastOverlay as unknown as { data?: { weapons?: WeaponDetection[] } })?.data?.weapons;
    if (incoming && incoming.length > 0) {
      return incoming.slice(0, 1);
    }
    if (heroOverlayFaces.length >= 2) {
      return [
        {
          id: 'weapon-demo',
          box: [420, 260, 520, 360],
          confidence: 0.87
        }
      ];
    }
    return [];
  }, [heroOverlayFaces.length, lastOverlay]);

  return (
    <div className={`live-ops live-ops-${mode}`}>
      <header className="live-ops-topbar">
        <div>
          <div className="live-ops-title">Live Ops Command Console</div>
          <div className="live-ops-subtitle">DSA Exhibition • Single Corridor Feed</div>
        </div>
        <div className="live-ops-modes">
          <button
            className={`mode-toggle ${mode === 'stealth' ? 'mode-active' : ''}`}
            onClick={() => setMode('stealth')}
          >
            Stealth Mission Mode
          </button>
          <button
            className={`mode-toggle ${mode === 'attraction' ? 'mode-active' : ''}`}
            onClick={() => setMode('attraction')}
          >
            Attraction Mode
          </button>
        </div>
      </header>

      <div className="live-ops-main">
        <aside className="live-ops-sidebar">
          <section className="panel identity-panel">
            <div className={`threat-band threat-${threatTone}`}>
              <span>{threatLabel}</span>
              <span className="threat-dot" />
            </div>
            <div className="identity-body">
              <div className="identity-faces">
                <div className={`face-frame face-live ${hasMatch ? 'face-match' : ''}`}>
                  <div className="face-label">Live Face</div>
                  <div className="face-placeholder">LIVE</div>
                </div>
                <div className={`face-frame face-ref ${hasMatch ? 'face-match' : ''} ${refPulse ? 'ref-pulse' : ''}`}>
                  <div className="face-label">Ref Face</div>
                  {referenceImage ? (
                    <img className="face-image" src={referenceImage} alt={watchlistLabel} />
                  ) : (
                    <div className="face-placeholder">REF</div>
                  )}
                </div>
              </div>
              <div className="identity-meta">
                <div className="identity-name">{primaryFace?.label ?? 'UNKNOWN SUBJECT'}</div>
                <div className="identity-alias">Alias: {watchlistFaces[0]?.label ?? '—'}</div>
                <div className="identity-score">Match Score: {primaryFace?.score ? `${(primaryFace.score * 100).toFixed(0)}%` : '—'}</div>
              </div>
              <div className="identity-actions">
                <button className="btn btn-confirm">Confirm</button>
                <button className="btn btn-reject">Reject</button>
                <button className="btn btn-escalate">Escalate</button>
              </div>
            </div>
          </section>

          <section className="panel timeline-panel">
            <div className="panel-title">Event Timeline</div>
            <div className="timeline-list">
              {timelineEvents.map((event) => (
                <div
                  key={event.id}
                  className={`timeline-item timeline-${event.severity} ${event.isMatch ? 'timeline-match' : ''}`}
                >
                  <div>
                    <div className="timeline-title">{event.label}</div>
                    <div className="timeline-detail">{event.detail}</div>
                  </div>
                  <div className="timeline-time">{event.time}</div>
                </div>
              ))}
            </div>
          </section>

          <section className="panel status-panel">
            <div className="panel-title">Camera Status</div>
            <div className="camera-status">
              <span className="status-dot" />
              <span>Cam01 Corridor (LIVE)</span>
            </div>
            <div className="camera-metrics">
              <div>FPS: {health?.fps?.toFixed(1) ?? '—'}</div>
              <div>Latency: {health?.latency_ms?.toFixed(0) ?? '—'} ms</div>
            </div>
          </section>
        </aside>

        <section className="live-ops-hero">
          <div
            className={`hero-frame ${matchPulse ? 'hero-match' : ''}`}
            ref={viewportRef}
            onMouseEnter={() => setHeroFocus(true)}
            onMouseLeave={() => setHeroFocus(false)}
          >
            {heroCamera?.enabled ? (
              <WebRTCPlayer
                cameraId={heroCamera.id}
                fallbackSrc={`/api/cameras/${heroCamera.id}/mjpeg`}
                className="hero-video"
                onFrameDimensions={(width, height) => setFrameDims({ width, height })}
              />
            ) : (
              <div className="hero-empty">No camera enabled. Add WEBCAM source 0.</div>
            )}

            <div className="hero-hud hero-hud-top">
              <div className="hud-left">
                <span className="hud-title">{heroName}</span>
                <span className={`live-indicator ${mode === 'attraction' ? 'live-pulse' : ''}`}>
                  LIVE
                </span>
                <span className="hud-metric">FPS {health?.fps?.toFixed(1) ?? '—'}</span>
              </div>
              <div className="hud-right">
                <span className="hud-chip">Mode: {mode === 'stealth' ? 'STEALTH' : 'ATTRACTION'}</span>
                <span className="hud-chip">Fusion Score: {fusionScore}</span>
              </div>
            </div>

            <div className="hero-hud hero-hud-bottom">
              <div className="hud-left">
                <span className="hud-title">
                  ID: {primaryFace?.label ?? 'UNKNOWN'}
                </span>
                <span className="hud-chip">Crowd: {hudMeta.crowdCount}</span>
                <span className="hud-chip">Restricted: {hudMeta.restrictedZone ? 'BREACH' : 'CLEAR'}</span>
                <span className="hud-chip">Loitering: {hudMeta.loiterTimer}</span>
              </div>
              <div className="hud-right">
                <span className="hud-chip">Weapon: {hudMeta.weaponDetected ? 'DETECTED' : 'NONE'}</span>
                <span className="hud-chip">{agentInsight}</span>
              </div>
            </div>

            <div className="overlay-layer">
              <div
                className={`restricted-zone ${hudMeta.restrictedZone ? 'zone-active' : ''}`}
                aria-hidden="true"
              />
              {visibleFaces.map((face, index) => {
                const [x1, y1, x2, y2] = face.box;
                const width = (x2 - x1) * overlayTransform.scale;
                const height = (y2 - y1) * overlayTransform.scale;
                const left = x1 * overlayTransform.scale + overlayTransform.offsetX;
                const top = y1 * overlayTransform.scale + overlayTransform.offsetY;
                const label = face.label.toLowerCase() === 'unknown'
                  ? 'UNKNOWN'
                  : `${face.label} ${(face.score * 100).toFixed(0)}%`;
                const isWatch = face.label.toLowerCase().includes('watchlist');
                return (
                  <div
                    key={`hero-face-${index}`}
                    className={`overlay-box ${isWatch ? 'overlay-watch' : ''} ${mode === 'attraction' ? 'overlay-scan' : ''}`}
                    style={{ left, top, width, height }}
                  >
                    <div className="overlay-label">{label}</div>
                  </div>
                );
              })}
              {weaponDetections.map((weapon) => {
                const [x1, y1, x2, y2] = weapon.box;
                const width = (x2 - x1) * overlayTransform.scale;
                const height = (y2 - y1) * overlayTransform.scale;
                const left = x1 * overlayTransform.scale + overlayTransform.offsetX;
                const top = y1 * overlayTransform.scale + overlayTransform.offsetY;
                return (
                  <div
                    key={weapon.id}
                    className="weapon-box"
                    style={{ left, top, width, height }}
                  >
                    <div className="weapon-label">WEAPON • {weapon.confidence.toFixed(2)}</div>
                  </div>
                );
              })}
              <div className="overlay-icon weapon-hook" aria-hidden="true">
                WEAPON
              </div>
            </div>
          </div>
        </section>
      </div>

      <footer className="live-ops-bottom">
        <div>Session: DSA-EXPO-01</div>
        <div>Cameras {status ? `${status.cameras_live} / ${status.cameras_total}` : '1 / 1'}</div>
        <div>Alerts: {watchlistFaces.length ? watchlistFaces.length : 0}</div>
        <div>Fusion Score: {fusionScore}</div>
        <div>Latency: {health?.latency_ms?.toFixed(0) ?? '—'} ms</div>
      </footer>
    </div>
  );
}
