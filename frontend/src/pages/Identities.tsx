import { useEffect, useState } from 'react';
import { createIdentity, enrollIdentityEmbedding, fetchIdentities, Identity } from '../api';

const emptyForm = {
  name: '',
  notes: ''
};

export default function Identities() {
  const [identities, setIdentities] = useState<Identity[]>([]);
  const [form, setForm] = useState(emptyForm);

  const load = async () => {
    try {
      const data = await fetchIdentities();
      setIdentities(data);
    } catch {
      setIdentities([]);
    }
  };

  useEffect(() => {
    load();
  }, []);

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    await createIdentity(form);
    setForm(emptyForm);
    load();
  };

  const handleEnroll = async (identityId: number) => {
    await enrollIdentityEmbedding(identityId);
    load();
  };

  return (
    <div className="space-y-6">
      <h1 className="text-2xl font-bold">Identities</h1>
      <div className="deftech-card">
        <form className="grid grid-cols-1 md:grid-cols-2 gap-4" onSubmit={handleSubmit}>
          <input
            className="bg-[#122454] border border-white/20 rounded px-2 py-1"
            placeholder="Identity Name"
            value={form.name}
            onChange={(event) => setForm({ ...form, name: event.target.value })}
            required
          />
          <input
            className="bg-[#122454] border border-white/20 rounded px-2 py-1"
            placeholder="Notes"
            value={form.notes}
            onChange={(event) => setForm({ ...form, notes: event.target.value })}
          />
          <button className="bg-green px-3 py-2 rounded" type="submit">
            Add Identity
          </button>
        </form>
      </div>

      <div className="deftech-card">
        <h2 className="text-lg font-semibold mb-2">Registered Identities</h2>
        <div className="space-y-2">
          {identities.length === 0 && <div className="text-white/60">No identities enrolled.</div>}
          {identities.map((identity) => (
            <div key={identity.id} className="flex items-center justify-between border-b border-white/10 py-2">
              <div>
                <div className="font-semibold">{identity.name}</div>
                <div className="text-xs text-white/70">{identity.notes || 'No notes'}</div>
                <div className="text-xs text-white/60">Embeddings: {identity.embedding_count}</div>
              </div>
              <button className="bg-amber text-black px-3 py-1 rounded" onClick={() => handleEnroll(identity.id)}>
                Enroll Demo Embedding
              </button>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
