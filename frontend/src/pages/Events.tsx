import { useState } from 'react';

const demoEvents = [
  { id: 1, camera: 'Camera 1', type: 'DEMO_EVENT', time: '2024-01-01T00:00:00Z', clip: '/static/event_clips/event_1_stub.mp4' }
];

export default function Events() {
  const [selectedClip, setSelectedClip] = useState<string | null>(null);

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Events</h1>
      <div className="deftech-card">
        <table className="w-full text-left text-sm">
          <thead className="text-white/70">
            <tr>
              <th className="pb-2">ID</th>
              <th className="pb-2">Camera</th>
              <th className="pb-2">Type</th>
              <th className="pb-2">Time</th>
              <th className="pb-2">Clip</th>
            </tr>
          </thead>
          <tbody>
            {demoEvents.map((event) => (
              <tr key={event.id} className="border-t border-white/10">
                <td className="py-2">{event.id}</td>
                <td className="py-2">{event.camera}</td>
                <td className="py-2">{event.type}</td>
                <td className="py-2 font-mono">{event.time}</td>
                <td className="py-2">
                  <button className="text-amber" onClick={() => setSelectedClip(event.clip)}>
                    View
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {selectedClip && (
        <div className="fixed inset-0 bg-black/80 flex items-center justify-center">
          <div className="bg-[#122454] p-4 rounded-lg w-3/4">
            <div className="flex justify-between items-center mb-2">
              <h2 className="text-lg font-semibold">Event Clip</h2>
              <button className="text-red" onClick={() => setSelectedClip(null)}>
                Close
              </button>
            </div>
            <video className="w-full" controls src={selectedClip} />
          </div>
        </div>
      )}
    </div>
  );
}
