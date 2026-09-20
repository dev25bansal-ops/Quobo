export type Mode = 'predictions' | 'demo-ground-truth'
export type ZoneMode = 'batch' | 'folder'

export interface ZoneRow {
  zone: string
  total: number
  clean: number
  dirty: number
  cleanliness: number
  psi: number
  band: string
  psi_ci: string
  psi_lo: number
  psi_hi: number
  dirty_rate_lo: number
  dirty_rate_hi: number
  vote_agreement: number | null
  counts: Record<string, number>
}

export interface Kpis {
  cleanliness: number | null
  worst_zone: { zone: string; psi: number; band: string } | null
  total_items: number
  zone_count: number
}

export interface BandDef {
  max: number
  name: string
  color: string
}

export interface Provenance {
  mode: string
  run_id?: string | null
  arm?: string | null
  clf?: string | null
  reps?: number | null
}

export interface DashboardData {
  generated_at: string
  mode: Mode
  zone_mode: ZoneMode
  provenance: Provenance
  source_label: string
  classes: string[]
  hazard_weights: Record<string, number>
  bands: BandDef[]
  band_colors: Record<string, string>
  kpis: Kpis
  zones: ZoneRow[]
}

export interface RunInfo {
  run_id: string
  status: string
  reps_completed: number
  arms: string[]
  clfs: string[]
  selectable: boolean
}
