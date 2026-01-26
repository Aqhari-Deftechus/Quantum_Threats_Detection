import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'Live Ops' },
  { to: '/events', label: 'Events' },
  { to: '/identities', label: 'Identities' },
  { to: '/cameras', label: 'Cameras' },
  { to: '/settings', label: 'Settings' }
];

export default function Nav() {
  return (
    <nav className="flex flex-col gap-2 p-4 bg-[#0b1533] min-h-screen">
      <div className="text-xl font-bold text-red">QTD</div>
      {links.map((link) => (
        <NavLink
          key={link.to}
          to={link.to}
          className={({ isActive }) =>
            `px-3 py-2 rounded-md ${isActive ? 'bg-red text-white' : 'text-white/80 hover:text-white'}`
          }
        >
          {link.label}
        </NavLink>
      ))}
    </nav>
  );
}
