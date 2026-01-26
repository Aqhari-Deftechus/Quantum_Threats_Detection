import { useEffect, useState } from 'react';
import ChatBox from '../components/ChatBox';
import { WsOverlay } from '../App';
import { fetchCameras, Camera } from '../api';

export default function LiveOps({ lastOverlay }: { lastOverlay: WsOverlay | null }) {
  const [cameras, setCameras] = useState<Camera[]>([]);

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

  const visibleCameras = cameras.slice(0, 4);
  const overlayFaces = lastOverlay?.data?.faces ?? [];
  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Live Ops</h1>
      <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
        {visibleCameras.length === 0 && (
          <div className="deftech-card h-56 flex items-center justify-center text-white/60">
            No cameras enabled. Add a WEBCAM source 0.
          </div>
        )}
        {visibleCameras.map((camera) => (
          <div key={camera.id} className="deftech-card h-56 flex flex-col justify-between">
            <div className="text-sm text-white/70">
              {camera.name} ({camera.source_type})
            </div>
            <div className="relative flex-1 w-full">
              {camera.enabled ? (
                <img
                  className="w-full h-full object-cover rounded"
                  src={`http://localhost:8000/api/cameras/${camera.id}/mjpeg`}
                  alt={`Camera ${camera.id}`}
                />
              ) : (
                <div className="flex-1 flex items-center justify-center text-white/50">
                  Camera disabled
                </div>
              )}
              {lastOverlay && lastOverlay.data.camera_id === camera.id && (
                <div className="absolute inset-0 pointer-events-none">
                  {overlayFaces.map((face: any, index: number) => {
                    const [x1, y1, x2, y2] = face.box;
                    const width = x2 - x1;
                    const height = y2 - y1;
                    return (
                      <div
                        key={`${camera.id}-${index}`}
                        className="absolute border-2 border-amber text-xs text-amber"
                        style={{ left: x1, top: y1, width, height }}
                      >
                        <div className="bg-black/60 px-1">
                          {face.label} {face.quality}
                        </div>
                      </div>
                    );
                  })}
                </div>
              )}
            </div>
          </div>
        ))}
      </div>
      <ChatBox />
    </div>
  );
}
