import { Map } from 'lucide-react'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import { cn } from '@/lib/utils'
import { badgeTextClass } from '@/lib/colors'
import type { DashboardData } from '@/types'

interface Props {
  data: DashboardData | null
  loading: boolean
  selectedZone: string | null
  onSelectZone: (z: string | null) => void
}

export function ZoneMap({ data, loading, selectedZone, onSelectZone }: Props) {
  if (loading && !data) return <Skeleton className="h-[240px] rounded-xl" />
  if (!data || data.zones.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Map className="size-4 text-primary" aria-hidden />
          Zone map — relative PSI
        </CardTitle>
        <CardDescription>
          Demo grid (TACO has no GPS). Click a cell to focus its row below; click again to
          clear.
        </CardDescription>
      </CardHeader>
      <CardContent>
        <div
          className="grid gap-3"
          style={{ gridTemplateColumns: `repeat(${Math.min(3, data.zones.length)}, minmax(0, 1fr))` }}
        >
          {data.zones.map((z) => {
            const color = data.band_colors[z.band] ?? '#475569'
            const selected = selectedZone === z.zone
            return (
              <button
                key={z.zone}
                type="button"
                aria-pressed={selected}
                onClick={() => onSelectZone(selected ? null : z.zone)}
                className={cn(
                  'group flex cursor-pointer flex-col items-center gap-0.5 rounded-lg px-3 py-5 transition-all duration-200',
                  badgeTextClass(z.band),
                  'hover:brightness-95 focus-visible:outline-none focus-visible:ring-2 focus-visible:ring-ring focus-visible:ring-offset-2',
                  selected && 'ring-2 ring-foreground ring-offset-2',
                )}
                style={{ backgroundColor: color }}
              >
                <span className="text-sm font-semibold">{z.zone}</span>
                <span className="font-mono text-xs tabular-nums opacity-90">
                  PSI {z.psi} · {z.total.toLocaleString()} items
                </span>
              </button>
            )
          })}
        </div>
        <div className="mt-4 flex flex-wrap items-center gap-x-4 gap-y-1.5 text-xs text-muted-foreground">
          {data.bands.map((b) => (
            <span key={b.name} className="inline-flex items-center gap-1.5">
              <span className="size-2.5 rounded-sm" style={{ backgroundColor: b.color }} />
              {b.name} ≤ {b.max}
            </span>
          ))}
        </div>
      </CardContent>
    </Card>
  )
}
