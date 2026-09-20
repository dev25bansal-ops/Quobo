import { useEffect, useMemo, useState } from 'react'
import { CircleSlash, TriangleAlert } from 'lucide-react'
import { Alert, AlertDescription, AlertTitle } from '@/components/ui/alert'
import { Empty, EmptyDescription, EmptyHeader, EmptyMedia, EmptyTitle } from '@/components/ui/empty'
import { Spinner } from '@/components/ui/spinner'
import { TooltipProvider } from '@/components/ui/tooltip'
import { useDashboard, useRuns } from '@/hooks/useDashboard'
import { HeaderBar } from '@/sections/HeaderBar'
import { KpiCards } from '@/sections/KpiCards'
import { ChartsPanel } from '@/sections/ChartsPanel'
import { ZoneMap } from '@/sections/ZoneMap'
import { ZoneTable } from '@/sections/ZoneTable'
import { MethodologyNotes } from '@/sections/MethodologyNotes'
import type { Mode, ZoneMode } from '@/types'

function App() {
  const [mode, setMode] = useState<Mode>('predictions')
  const [runId, setRunId] = useState<string | null>(null) // null = latest complete
  const [arm, setArm] = useState('D_qubo')
  const [clf, setClf] = useState('rbf_svm')
  const [zoneMode, setZoneMode] = useState<ZoneMode>('batch')
  const [live, setLive] = useState(true)
  const [selectedZone, setSelectedZone] = useState<string | null>(null)

  const { runs } = useRuns()
  const selectable = useMemo(() => runs.filter((r) => r.selectable), [runs])
  const activeRun = useMemo(
    () => selectable.find((r) => r.run_id === runId) ?? selectable[selectable.length - 1],
    [selectable, runId],
  )

  // Keep arm/clf legal for the selected run (fall back to preferred, then first).
  useEffect(() => {
    if (!activeRun?.arms?.length) return
    if (!activeRun.arms.includes(arm)) {
      setArm(activeRun.arms.includes('D_qubo') ? 'D_qubo' : activeRun.arms[0])
    }
  }, [activeRun, arm])
  useEffect(() => {
    if (!activeRun?.clfs?.length) return
    if (!activeRun.clfs.includes(clf)) {
      setClf(activeRun.clfs.includes('rbf_svm') ? 'rbf_svm' : activeRun.clfs[0])
    }
  }, [activeRun, clf])

  const params = useMemo(
    () => ({
      mode,
      runId: mode === 'predictions' ? activeRun?.run_id ?? null : null,
      arm,
      clf,
      zoneMode,
    }),
    [mode, activeRun?.run_id, arm, clf, zoneMode],
  )

  const { data, error, loading, updatedAt, refresh } = useDashboard(params, live)

  useEffect(() => setSelectedZone(null), [params.mode, params.zoneMode, activeRun?.run_id, arm, clf])

  return (
    <div className="min-h-screen bg-background font-sans">
      <HeaderBar
        mode={mode}
        onMode={setMode}
        runs={runs}
        runId={runId}
        onRunId={setRunId}
        arm={arm}
        onArm={setArm}
        clf={clf}
        onClf={setClf}
        zoneMode={zoneMode}
        onZoneMode={setZoneMode}
        live={live}
        onLive={setLive}
        onRefresh={() => void refresh()}
        loading={loading}
        updatedAt={updatedAt}
        sourceLabel={data?.source_label ?? null}
      />

      <main className="mx-auto max-w-7xl space-y-4 px-6 py-5">
        <TooltipProvider>
          {error && !data && (
            <Alert variant="destructive">
              <CircleSlash className="size-4" aria-hidden />
              <AlertTitle>No data — provenance check refused this request</AlertTitle>
              <AlertDescription>
                {error}
                <br />
                {mode === 'predictions' && (
                  <span className="mt-1 block text-xs">
                    There is no fallback by design. Finish an experiment run, pick another
                    verified run, or switch the source to “Demo ground truth”.
                  </span>
                )}
              </AlertDescription>
            </Alert>
          )}

          {error && data && (
            <div className="flex items-center gap-2 rounded-lg border border-destructive/40 bg-destructive/5 px-3 py-2 text-xs text-destructive">
              <TriangleAlert className="size-3.5 shrink-0" aria-hidden />
              Live refresh failed: {error} — showing last good data
              {updatedAt ? ` (${updatedAt.toLocaleTimeString()})` : ''}
            </div>
          )}

          {loading && !data ? (
            <div className="flex items-center justify-center gap-2 py-24 text-sm text-muted-foreground">
              <Spinner className="size-4" /> loading zone analytics…
            </div>
          ) : data && data.zones.length === 0 ? (
            <Empty>
              <EmptyHeader>
                <EmptyMedia variant="icon">
                  <CircleSlash className="size-6" aria-hidden />
                </EmptyMedia>
                <EmptyTitle>No observations</EmptyTitle>
                <EmptyDescription>
                  No classified items were available for this selection, so no zone metrics
                  were computed. Source: {data.source_label}
                </EmptyDescription>
              </EmptyHeader>
            </Empty>
          ) : (
            <>
              <KpiCards data={data} loading={loading} />
              <ChartsPanel
                data={data}
                loading={loading}
                selectedZone={selectedZone}
                onSelectZone={setSelectedZone}
              />
              <ZoneMap
                data={data}
                loading={loading}
                selectedZone={selectedZone}
                onSelectZone={setSelectedZone}
              />
              <ZoneTable
                data={data}
                loading={loading}
                selectedZone={selectedZone}
                onSelectZone={setSelectedZone}
              />
              <MethodologyNotes data={data} />
            </>
          )}
        </TooltipProvider>
      </main>
    </div>
  )
}

export default App
