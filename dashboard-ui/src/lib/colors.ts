export const CLASS_COLORS: Record<string, string> = {
  cigarette: '#b91c1c',
  bottle: '#1d4ed8',
  can: '#0891b2',
  carton: '#65a30d',
  cup: '#d97706',
  lid: '#7c3aed',
}

const FALLBACK_COLORS = ['#475569', '#334155', '#0f766e', '#a21caf', '#2563eb', '#ca8a04']

export function classColor(cls: string, idx = 0): string {
  return CLASS_COLORS[cls] ?? FALLBACK_COLORS[idx % FALLBACK_COLORS.length]
}

/** Bands whose EPA-style colors are too light for white badge text. */
const DARK_TEXT_BANDS = new Set(['Good', 'Moderate'])

export function badgeTextClass(band: string): string {
  return DARK_TEXT_BANDS.has(band) ? 'text-slate-900' : 'text-white'
}
