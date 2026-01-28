import { NavLink } from 'react-router-dom';

const links = [
  { to: '/', label: 'Live Ops', icon: '⬤' },
  { to: '/events', label: 'Threat Timeline', icon: '▣' },
  { to: '/identities', label: 'Persons of Interest', icon: '◆' },
  { to: '/cameras', label: 'Cameras', icon: '▥' },
  { to: '/settings', label: 'Settings', icon: '⚙' }
];

export default function Nav() {
  return (
    <nav className="nav-rail">
      <div className="nav-title">
        <span className="nav-mark">QATRS</span>
        <span className="nav-subtitle">Quantum Adaptive Threat Recognition</span>
      </div>
      <div className="nav-section">TACTICAL</div>
      <div className="nav-links">
        {links.map((link) => (
          <NavLink
            key={link.to}
            to={link.to}
            className={({ isActive }) =>
              `nav-link ${isActive ? 'nav-link-active' : ''}`
            }
          >
            <span className="nav-icon">{link.icon}</span>
            <span className="nav-label">{link.label}</span>
          </NavLink>
        ))}
      </div>
    </nav>
  );
}
