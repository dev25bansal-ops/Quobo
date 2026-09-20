import { Activity, AlertTriangle, Droplets, RotateCw } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import { Button } from '@/components/ui/button'
import { Label } from '@/components/ui/label'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { Switch } from '@/components/ui/switch'
import { ToggleGroup, ToggleGroupItem } from '@/components/ui/toggle-group'
import type { Mode, RunInfo, ZoneMode } from '@/types'

interface Props {
  mode: Mode
  onMode: (m: Mode) => void
  runs: RunInfo[]
  runId: string | null
  onRunId: (r: string | null) => void
  arm: string
  onArm: (a: string) => void
  clf: string
  onClf: (c: string) => void
  zoneMode: ZoneMode
  onZoneMode: (z: ZoneMode) => void
  live: boolean
  onLive: (v: boolean) => void
  onRefresh: () => void
  loading: boolean
  updatedAt: Date | null
  sourceLabel: string | null
}

const LATEST = '__latest__'

export function HeaderBar(p: Props) {
  const selectable = p.runs.filter((r) => r.selectable).slice().reverse() // newest first
  const truth = p.mode === 'demo-ground-truth'
  const activeRun =
    selectable.find((r) => r.run_id === p.runId) ?? selectable[0]

  return (
    <header className="sticky top-0 z-30 border-b bg-card/95 backdrop-blur supports-[backdrop-filter]:bg-card/80">
      <div className="mx-auto flex max-w-7xl flex-wrap items-center gap-x-6 gap-y-3 px-6 py-3">
        <div className="flex min-w-0 items-center gap-3">
          <div className="flex size-10 shrink-0 items-center justify-center rounded-lg bg-primary text-primary-foreground">
            <Droplets className="size-5" aria-hidden />
          </div>
          <div className="min-w-0">
            <h1 className="truncate text-lg font-semibold leading-tight tracking-tight">
              Quobo Pollution Dashboard
            </h1>
            <p className="truncate text-xs text-muted-foreground" title={p.sourceLabel ?? ''}>
              {p.sourceLabel ?? 'connecting…'}
            </p>
          </div>
        </div>

        <div className="ml-auto flex flex-wrap items-center gap-2">
          <div className="flex items-center gap-1.5">
            <Label htmlFor="mode-sel" className="sr-only">
              Data source
            </Label>
            <Select value={p.mode} onValueChange={(v) => p.onMode(v as Mode)}>
              <SelectTrigger id="mode-sel" size="sm" className="w-[13.5rem] cursor-pointer">
                <SelectValue placeholder="Data source" />
              </SelectTrigger>
              <SelectContent>
                <SelectItem value="predictions">Classifier predictions</SelectItem>
                <SelectItem value="demo-ground-truth">Demo ground truth (no model)</SelectItem>
              </SelectContent>
            </Select>
          </div>

          {!truth && (
            <>
              <div className="flex items-center gap-1.5">
                <Label htmlFor="run-sel" className="sr-only">
                  Experiment run
                </Label>
                <Select
                  value={p.runId ?? LATEST}
                  onValueChange={(v) => p.onRunId(v === LATEST ? null : v)}
                >
                  <SelectTrigger id="run-sel" size="sm" className="w-[15.5rem] cursor-pointer">
                    <SelectValue placeholder="Run" />
                  </SelectTrigger>
                  <SelectContent>
                    <SelectItem value={LATEST}>Latest complete run</SelectItem>
                    {selectable.map((r) => (
                      <SelectItem key={r.run_id} value={r.run_id}>
                        {r.run_id} · {r.reps_completed} reps
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-1.5">
                <Label htmlFor="arm-sel" className="sr-only">
                  Selection arm
                </Label>
                <Select value={p.arm} onValueChange={p.onArm}>
                  <SelectTrigger id="arm-sel" size="sm" className="w-[8.5rem] cursor-pointer">
                    <SelectValue placeholder="Arm" />
                  </SelectTrigger>
                  <SelectContent>
                    {(activeRun?.arms?.length ? activeRun.arms : [p.arm]).map((a) => (
                      <SelectItem key={a} value={a}>
                        {a}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
              <div className="flex items-center gap-1.5">
                <Label htmlFor="clf-sel" className="sr-only">
                  Classifier
                </Label>
                <Select value={p.clf} onValueChange={p.onClf}>
                  <SelectTrigger id="clf-sel" size="sm" className="w-[9rem] cursor-pointer">
                    <SelectValue placeholder="Classifier" />
                  </SelectTrigger>
                  <SelectContent>
                    {(activeRun?.clfs?.length ? activeRun.clfs : [p.clf]).map((c) => (
                      <SelectItem key={c} value={c}>
                        {c}
                      </SelectItem>
                    ))}
                  </SelectContent>
                </Select>
              </div>
            </>
          )}

          <ToggleGroup
            type="single"
            value={p.zoneMode}
            onValueChange={(v) => v && p.onZoneMode(v as ZoneMode)}
            variant="outline"
            size="sm"
            aria-label="Zone assignment mode"
          >
            <ToggleGroupItem value="batch" className="cursor-pointer">
              batch zones
            </ToggleGroupItem>
            <ToggleGroupItem value="folder" className="cursor-pointer">
              geo folders
            </ToggleGroupItem>
          </ToggleGroup>

          <div className="flex items-center gap-2 rounded-md border px-2.5 py-1.5">
            <Activity
              className={`size-4 ${p.live ? 'text-emerald-600' : 'text-muted-foreground'}`}
              aria-hidden
            />
            <Switch
              id="live-switch"
              checked={p.live}
              onCheckedChange={p.onLive}
              className="cursor-pointer"
            />
            <Label htmlFor="live-switch" className="cursor-pointer text-xs font-medium">
              Live
            </Label>
          </div>

          <Button
            variant="outline"
            size="sm"
            onClick={() => void p.onRefresh()}
            disabled={p.loading}
            className="cursor-pointer"
            aria-label="Refresh now"
          >
            <RotateCw className={`size-4 ${p.loading ? 'animate-spin' : ''}`} aria-hidden />
          </Button>
        </div>
      </div>

      <div className="border-t bg-accent/15">
        <div className="mx-auto flex max-w-7xl items-center gap-2 px-6 py-1.5 text-xs text-accent-foreground">
          <AlertTriangle className="size-3.5 shrink-0" aria-hidden />
          <span>
            <b>PSI is a project-defined relative index — unvalidated, not a health or
            air-quality standard.</b>
            {p.zoneMode === 'batch'
              ? ' Zone assignment is a demo mapping of TACO image batches (no GPS metadata).'
              : ' Zone assignment reads data/geo/<Zone>/<class>/ folder layout.'}
          </span>
          {p.updatedAt && (
            <Badge variant="secondary" className="ml-auto shrink-0 font-mono tabular-nums">
              updated {p.updatedAt.toLocaleTimeString()}
            </Badge>
          )}
        </div>
      </div>
    </header>
  )
}
