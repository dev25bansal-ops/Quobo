import { useCallback, useEffect, useRef, useState } from 'react'
import { fetchDashboard, fetchRuns, type DashParams } from '@/lib/api'
import type { DashboardData, RunInfo } from '@/types'

const REFRESH_MS = 15_000
const RUNS_REFRESH_MS = 60_000

/** Live dashboard data: refetches on param change, on an interval while
 *  `live` is on (only when the tab is visible), and never drops the last
 *  good payload on a transient refresh error. */
export function useDashboard(params: DashParams, live: boolean) {
  const [data, setData] = useState<DashboardData | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [loading, setLoading] = useState(true)
  const [updatedAt, setUpdatedAt] = useState<Date | null>(null)
  const abortRef = useRef<AbortController | null>(null)

  const load = useCallback(async () => {
    abortRef.current?.abort()
    const ctrl = new AbortController()
    abortRef.current = ctrl
    try {
      const d = await fetchDashboard(params, ctrl.signal)
      if (!ctrl.signal.aborted) {
        setData(d)
        setError(null)
        setUpdatedAt(new Date())
      }
    } catch (e) {
      if ((e as Error)?.name === 'AbortError') return
      setError((e as Error).message || 'request failed')
    } finally {
      if (abortRef.current === ctrl) setLoading(false)
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [params.mode, params.runId, params.arm, params.clf, params.zoneMode])

  useEffect(() => {
    setLoading(true)
    void load()
    return () => abortRef.current?.abort()
  }, [load])

  useEffect(() => {
    if (!live) return
    const t = setInterval(() => {
      if (document.visibilityState === 'visible') void load()
    }, REFRESH_MS)
    const onVis = () => {
      if (document.visibilityState === 'visible') void load()
    }
    document.addEventListener('visibilitychange', onVis)
    return () => {
      clearInterval(t)
      document.removeEventListener('visibilitychange', onVis)
    }
  }, [live, load])

  return { data, error, loading, updatedAt, refresh: load }
}

/** Experiment runs, polled so a paper run that just finished appears live. */
export function useRuns() {
  const [runs, setRuns] = useState<RunInfo[]>([])
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let stop = false
    const tick = () =>
      fetchRuns()
        .then((r) => {
          if (!stop) {
            setRuns(r)
            setError(null)
          }
        })
        .catch((e: Error) => {
          if (!stop) setError(e.message)
        })
    tick()
    const t = setInterval(tick, RUNS_REFRESH_MS)
    return () => {
      stop = true
      clearInterval(t)
    }
  }, [])

  return { runs, error }
}
