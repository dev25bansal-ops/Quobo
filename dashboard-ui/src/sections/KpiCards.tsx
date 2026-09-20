import { Boxes, Gauge, Layers, Sparkles, Trash2 } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import type { DashboardData } from '@/types'

interface Props {
  data: DashboardData | null
  loading: boolean
}

function Kpi({
  icon,
  label,
  value,
  sub,
  valueClass,
}: {
  icon: React.ReactNode
  label: string
  value: React.ReactNode
  sub?: string
  valueClass?: string
}) {
  return (
    <Card className="gap-0 py-4">
      <CardHeader className="px-4 pb-0">
        <CardDescription className="flex items-center gap-1.5 text-[11px] font-medium uppercase tracking-wider">
          {icon}
          {label}
        </CardDescription>
      </CardHeader>
      <CardContent className="px-4">
        <div className={`font-mono text-3xl font-semibold tabular-nums leading-tight ${valueClass ?? ''}`}>
          {value}
        </div>
        {sub && <p className="mt-1 text-xs text-muted-foreground">{sub}</p>}
      </CardContent>
    </Card>
  )
}

export function KpiCards({ data, loading }: Props) {
  if (loading && !data) {
    return (
      <div className="grid grid-cols-2 gap-4 md:grid-cols-4 xl:grid-cols-5">
        {Array.from({ length: 5 }).map((_, i) => (
          <Skeleton key={i} className="h-[104px] rounded-xl" />
        ))}
      </div>
    )
  }
  if (!data) return null
  const k = data.kpis
  const bandColor = data.band_colors[k.worst_zone?.band ?? ''] ?? '#475569'
  const agreements = data.zones.filter((z) => z.vote_agreement != null)
  const avgAgreement =
    agreements.length > 0
      ? agreements.reduce((s, z) => s + (z.vote_agreement ?? 0) * z.total, 0) /
        agreements.reduce((s, z) => s + z.total, 0)
      : null

  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-4 xl:grid-cols-5">
      <Kpi
        icon={<Sparkles className="size-3.5" aria-hidden />}
        label="City cleanliness"
        value={k.cleanliness == null ? '—' : `${k.cleanliness}%`}
        sub="clean / total items (project formula)"
      />
      <Kpi
        icon={<Gauge className="size-3.5" aria-hidden />}
        label="Worst-zone PSI"
        value={
          k.worst_zone ? (
            <span style={{ color: bandColor }}>{k.worst_zone.psi}</span>
          ) : (
            '—'
          )
        }
        sub={
          k.worst_zone
            ? `${k.worst_zone.zone} · ${k.worst_zone.band} (relative, unvalidated)`
            : 'no observations'
        }
      />
      <Kpi
        icon={<Trash2 className="size-3.5" aria-hidden />}
        label="Classified items"
        value={k.total_items.toLocaleString()}
        sub={data.mode === 'predictions' ? 'held-out crops, majority vote' : 'ground-truth crops (demo)'}
      />
      <Kpi
        icon={<Layers className="size-3.5" aria-hidden />}
        label="Zones monitored"
        value={k.zone_count}
        sub={data.zone_mode === 'batch' ? 'demo batch mapping' : 'geo folder layout'}
      />
      {avgAgreement != null && (
        <Kpi
          icon={<Boxes className="size-3.5" aria-hidden />}
          label="Vote agreement"
          value={`${Math.round(avgAgreement * 100)}%`}
          sub={`repeat-to-repeat stability over ${data.provenance.reps ?? '?'} reps (not accuracy)`}
        />
      )}
    </div>
  )
}
