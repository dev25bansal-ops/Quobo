import type { Mode, ZoneMode } from '@/types'
import type { DashboardData, RunInfo } from '@/types'

async function getJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  const res = await fetch(url, { signal })
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`
    try {
      const body = await res.json()
      if (typeof body?.detail === 'string') detail = body.detail
    } catch {
      /* body was not JSON — keep the status line */
    }
    throw new Error(detail)
  }
  return (await res.json()) as T
}

export interface DashParams {
  mode: Mode
  runId: string | null
  arm: string
  clf: string
  zoneMode: ZoneMode
}

export function fetchRuns(signal?: AbortSignal): Promise<RunInfo[]> {
  return getJson<RunInfo[]>('/api/runs', signal)
}

export function fetchDashboard(p: DashParams, signal?: AbortSignal): Promise<DashboardData> {
  const q = new URLSearchParams({
    mode: p.mode,
    arm: p.arm,
    clf: p.clf,
    zone_mode: p.zoneMode,
  })
  if (p.runId) q.set('run_id', p.runId)
  return getJson<DashboardData>(`/api/dashboard?${q.toString()}`, signal)
}
