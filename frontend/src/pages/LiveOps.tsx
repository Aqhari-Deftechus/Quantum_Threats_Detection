import { useEffect, useMemo, useState } from 'react';
import ChatBox from '../components/ChatBox';
import { WsOverlay } from '../App';
import { fetchCameraHealth, fetchCameras, fetchStatus, Camera, CameraHealth, StatusResponse } from '../api';

type ThreatFeedItem = {
  id: string;
  module: '[FACE]' | '[WEAPON]' | '[FUSION]';
  severity: 'INFO' | 'WARN' | 'HIGH' | 'CRITICAL';
  timestamp: string;
  cameraId: number;
  text: string;
  subjectLabel: string;
  poiLabel?: string;
  reasonCodes?: string[];
  confidence?: string;
  weaponType?: string;
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

export default function LiveOps({ lastOverlay }: { lastOverlay: WsOverlay | null }) {
  const [cameras, setCameras] = useState<Camera[]>([]);
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [cameraHealth, setCameraHealth] = useState<Record<number, CameraHealth>>({});
  const [focusedCameraId, setFocusedCameraId] = useState<number | null>(null);
  const [selectedFeedItem, setSelectedFeedItem] = useState<ThreatFeedItem | null>(null);
  const [policyOpen, setPolicyOpen] = useState(false);
  const [fullscreenCameraId, setFullscreenCameraId] = useState<number | null>(null);

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
          cameras.slice(0, 4).map(async (camera) => [camera.id, await fetchCameraHealth(camera.id)] as const)
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

  const visibleCameras = cameras.slice(0, 4);
  const overlayFaces = lastOverlay?.data?.faces ?? [];
  const confirmedFaces = overlayFaces.filter(
    (face) => face.label && face.label.trim().toLowerCase() !== 'unknown'
  );
  const watchlistFaces = overlayFaces.filter((face) => face.label.toLowerCase().includes('watchlist'));

  const threatFeedItems = useMemo(() => {
    const items: ThreatFeedItem[] = [];
    if (lastOverlay?.data?.camera_id && overlayFaces.length > 0) {
      overlayFaces.forEach((face, index) => {
        const isWatchlist = face.label.toLowerCase().includes('watchlist');
        items.push({
          id: `face-${lastOverlay.data.camera_id}-${index}`,
          module: '[FACE]',
          severity: isWatchlist ? 'HIGH' : 'WARN',
          timestamp: lastOverlay.timestamp,
          cameraId: lastOverlay.data.camera_id,
          text: isWatchlist ? 'Watchlist match detected' : 'Face detected - verification pending',
          subjectLabel: face.label || 'UNKNOWN',
          poiLabel: isWatchlist ? 'POI-0417 / GHOST FOX' : 'UNKNOWN',
          reasonCodes: [
            isWatchlist ? 'MATCH_HIT' : 'FACE_ONLY',
            face.quality ? `QUALITY_${face.quality.toUpperCase()}` : 'QUALITY_UNKNOWN'
          ],
          confidence: face.score ? face.score.toFixed(2) : '—'
        });
      });
    }

    if (items.length === 0) {
      items.push(
        {
          id: 'fusion-idle',
          module: '[FUSION]',
          severity: 'INFO',
          timestamp: new Date().toISOString(),
          cameraId: visibleCameras[0]?.id ?? 0,
          text: 'No correlated threats. Monitoring remains active.',
          subjectLabel: 'NORMAL',
          reasonCodes: ['NO_MATCH']
        },
        {
          id: 'weapon-placeholder',
          module: '[WEAPON]',
          severity: 'WARN',
          timestamp: new Date().toISOString(),
          cameraId: visibleCameras[0]?.id ?? 0,
          text: 'Weapon module standing by',
          subjectLabel: 'NOT_VISIBLE',
          weaponType: '—'
        }
      );
    }

    return items;
  }, [lastOverlay, overlayFaces, visibleCameras]);

  const threatPosture = useMemo(() => {
    const hasWatchlist = watchlistFaces.length > 0;
    const hasWeapon = false;
    if (hasWatchlist && hasWeapon) return 'CRITICAL';
    if (hasWatchlist) return 'ELEVATED';
    if (hasWeapon) return 'HIGH';
    return 'NORMAL';
  }, [watchlistFaces]);

  const autoOpenChat = threatPosture === 'CRITICAL';

  const activeContext = selectedFeedItem
    ? {
        id: selectedFeedItem.id,
        label: `CAM-${selectedFeedItem.cameraId}`,
        type: 'alert' as const,
        severity: selectedFeedItem.severity
      }
    : focusedCameraId
      ? {
          id: `camera-${focusedCameraId}`,
          label: `CAM-${focusedCameraId}`,
          type: 'camera' as const
        }
      : null;

  return (
    <div className="live-ops">
      <div className="live-ops-center">
        <header className="live-ops-header">
          <div>
            <div className="live-ops-title">Live Ops Command Console</div>
            <div className="live-ops-subtitle">Real-time tactical monitoring • Operator authority retained</div>
          </div>
          <div className="live-ops-threat">
            <span className={`threat-pill threat-${threatPosture.toLowerCase()}`}>{threatPosture}</span>
            <span className="threat-pill threat-neutral">FUSION POSTURE</span>
          </div>
        </header>

        <section className="video-wall">
          {visibleCameras.length === 0 && (
            <div className="video-tile video-tile-empty">
              No cameras enabled. Add a WEBCAM source 0.
            </div>
          )}
          {visibleCameras.map((camera) => {
            const isFocused = focusedCameraId === camera.id;
            const health = cameraHealth[camera.id];
            return (
              <div
                key={camera.id}
                className={`video-tile ${isFocused ? 'video-tile-focused' : ''}`}
                onClick={() => setFocusedCameraId(camera.id)}
              >
                <div className="video-tile-header">
                  <div>
                    <div className="video-tile-label">{camera.name}</div>
                    <div className="video-tile-meta">
                      CAM-{camera.id} • {camera.source_type}
                    </div>
                  </div>
                  <div className="video-tile-health">
                    <span className={`health-dot ${health?.status === 'UP' ? 'health-up' : 'health-down'}`} />
                    {health?.status ?? '—'}
                  </div>
                </div>
                <div className="video-frame">
                  {camera.enabled ? (
                    <img
                      className="video-feed"
                      src={`http://localhost:8000/api/cameras/${camera.id}/mjpeg`}
                      alt={`Camera ${camera.id}`}
                    />
                  ) : (
                    <div className="video-disabled">Camera disabled</div>
                  )}
                  {lastOverlay && lastOverlay.data.camera_id === camera.id && (
                    <div className="overlay-layer">
                      {confirmedFaces.map((face, index) => {
                        const [x1, y1, x2, y2] = face.box;
                        const width = x2 - x1;
                        const height = y2 - y1;
                        const label = `[FACE] T${index + 1} | ${face.label} | Q:${face.quality} | SIM ${face.score.toFixed(2)}`;
                        return (
                          <div
                            key={`${camera.id}-${index}`}
                            className="overlay-box"
                            style={{ left: x1, top: y1, width, height }}
                          >
                            <div className="overlay-label">{label}</div>
                          </div>
                        );
                      })}
                    </div>
                  )}
                </div>
                <div className="video-tile-footer">
                  <button
                    className="video-action"
                    onClick={(event) => {
                      event.stopPropagation();
                      setFullscreenCameraId(camera.id);
                    }}
                  >
                    Fullscreen
                  </button>
                  <div className="video-stats">
                    FPS {health?.fps?.toFixed(1) ?? '—'} • Lat {health?.latency_ms?.toFixed(0) ?? '—'} ms
                  </div>
                </div>
              </div>
            );
          })}
        </section>
      </div>

      <aside className="intel-rail">
        <section className="intel-card">
          <div className="intel-card-title">Threat Snapshot</div>
          <div className="intel-kpi-grid">
            <div>
              <div className="intel-kpi-label">Threat Posture</div>
              <div className="intel-kpi-value">{threatPosture}</div>
            </div>
            <div>
              <div className="intel-kpi-label">Watchlist Detected</div>
              <div className="intel-kpi-value">
                {lastOverlay ? watchlistFaces.length : '—'}
              </div>
            </div>
            <div>
              <div className="intel-kpi-label">Weapons Visible</div>
              <div className="intel-kpi-value">—</div>
            </div>
            <div>
              <div className="intel-kpi-label">Cameras Live / Total</div>
              <div className="intel-kpi-value">
                {status ? `${status.cameras_live} / ${status.cameras_total}` : '—'}
              </div>
            </div>
            <div>
              <div className="intel-kpi-label">P95 Latency</div>
              <div className="intel-kpi-value">—</div>
            </div>
          </div>
        </section>

        <section className="intel-card intel-feed">
          <div className="intel-card-title">Threat Feed (A.D.A.M)</div>
          <div className="feed-list">
            {threatFeedItems.map((item) => (
              <button
                key={item.id}
                className={`feed-item ${selectedFeedItem?.id === item.id ? 'feed-item-active' : ''}`}
                onClick={() => {
                  setSelectedFeedItem(item);
                  setFocusedCameraId(item.cameraId);
                }}
              >
                <div className="feed-item-head">
                  <span className={`feed-tag feed-${item.severity.toLowerCase()}`}>{item.severity}</span>
                  <span className="feed-module">{item.module}</span>
                  <span className="feed-time">{new Date(item.timestamp).toLocaleTimeString()}</span>
                </div>
                <div className="feed-item-body">
                  <div className="feed-item-main">{item.text}</div>
                  <div className="feed-item-meta">
                    CAM-{item.cameraId} • {item.subjectLabel}
                  </div>
                </div>
              </button>
            ))}
          </div>
        </section>

        <section className="intel-card intel-details">
          <div className="intel-card-title">Case Details (M.A.R.Y.A.M)</div>
          {selectedFeedItem ? (
            <div className="details-content">
              <div className="details-header">
                <div className="details-thumb">
                  <span>POI</span>
                </div>
                <div>
                  <div className="details-title">{selectedFeedItem.poiLabel ?? 'UNKNOWN'}</div>
                  <div className="details-meta">{selectedFeedItem.subjectLabel}</div>
                </div>
              </div>
              <div className="details-grid">
                <div>
                  <div className="details-label">Confidence</div>
                  <div className="details-value">{selectedFeedItem.confidence ?? '—'}</div>
                </div>
                <div>
                  <div className="details-label">Stability</div>
                  <div className="details-value">WINDOW 3/5</div>
                </div>
                <div>
                  <div className="details-label">Reason Codes</div>
                  <div className="details-value">
                    {selectedFeedItem.reasonCodes?.join(', ') ?? '—'}
                  </div>
                </div>
                <div>
                  <div className="details-label">Evidence</div>
                  <div className="details-value link">clip://pending</div>
                </div>
              </div>
              <button className="details-policy" onClick={() => setPolicyOpen((prev) => !prev)}>
                {policyOpen ? 'Hide' : 'Show'} policy explanation
              </button>
              {policyOpen && (
                <div className="details-policy-body">
                  Fusion rules applied. Operator retains final authority. No autonomous action taken.
                </div>
              )}
            </div>
          ) : (
            <div className="details-empty">Select a feed item to review case details.</div>
          )}
        </section>
      </aside>

      {fullscreenCameraId && (
        <div className="fullscreen-modal" onClick={() => setFullscreenCameraId(null)}>
          <div className="fullscreen-panel" onClick={(event) => event.stopPropagation()}>
            <div className="fullscreen-header">
              Camera {fullscreenCameraId} • Fullscreen
              <button className="fullscreen-close" onClick={() => setFullscreenCameraId(null)}>
                Close
              </button>
            </div>
            <div className="fullscreen-body">
              <img
                className="fullscreen-feed"
                src={`http://localhost:8000/api/cameras/${fullscreenCameraId}/mjpeg`}
                alt={`Camera ${fullscreenCameraId}`}
              />
            </div>
          </div>
        </div>
      )}

      <ChatBox context={activeContext} autoOpen={autoOpenChat} />
    </div>
  );
}
