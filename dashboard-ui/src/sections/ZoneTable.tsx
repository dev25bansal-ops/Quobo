import { Table } from 'lucide-react'
import { Badge } from '@/components/ui/badge'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Progress } from '@/components/ui/progress'
import { Skeleton } from '@/components/ui/skeleton'
import {
  Table as ShTable,
  TableBody,
  TableCell,
  TableHead,
  TableHeader,
  TableRow,
} from '@/components/ui/table'
import {
  Tooltip,
  TooltipContent,
  TooltipTrigger,
} from '@/components/ui/tooltip'
import { badgeTextClass } from '@/lib/colors'
import { cn } from '@/lib/utils'
import type { DashboardData } from '@/types'

interface Props {
  data: DashboardData | null
  loading: boolean
  selectedZone: string | null
  onSelectZone: (z: string | null) => void
}

export function ZoneTable({ data, loading, selectedZone, onSelectZone }: Props) {
  if (loading && !data) return <Skeleton className="h-[280px] rounded-xl" />
  if (!data || data.zones.length === 0) return null

  return (
    <Card>
      <CardHeader>
        <CardTitle className="flex items-center gap-2 text-base">
          <Table className="size-4 text-primary" aria-hidden />
          Zone detail
        </CardTitle>
        <CardDescription>
          PSI range = conditional sampling bounds (Bonferroni-adjusted Wilson intervals,
          normalization reference fixed). Dirty-rate range = 95% Wilson interval on
          dirty/total. Vote agreement = repeat-to-repeat stability, not accuracy.
        </CardDescription>
      </CardHeader>
      <CardContent className="px-0">
        <ShTable>
          <TableHeader>
            <TableRow>
              <TableHead>Zone</TableHead>
              <TableHead className="text-right">Relative PSI</TableHead>
              <TableHead className="text-right">Band</TableHead>
              <TableHead className="w-[160px]">Cleanliness</TableHead>
              <TableHead className="text-right">Items</TableHead>
              <TableHead className="text-right">Clean</TableHead>
              <TableHead className="text-right">
                Dirty{' '}
                <span className="font-normal text-muted-foreground">(rate 95% CI)</span>
              </TableHead>
              <TableHead className="text-right">Vote agree.</TableHead>
            </TableRow>
          </TableHeader>
          <TableBody>
            {data.zones.map((z) => {
              const selected = selectedZone === z.zone
              const bandColor = data.band_colors[z.band] ?? '#475569'
              return (
                <TableRow
                  key={z.zone}
                  onClick={() => onSelectZone(selected ? null : z.zone)}
                  className={cn(
                    'cursor-pointer',
                    selected && 'bg-primary/5 hover:bg-primary/10',
                  )}
                >
                  <TableCell className="font-medium">{z.zone}</TableCell>
                  <TableCell className="text-right">
                    <Tooltip>
                      <TooltipTrigger asChild>
                        <span className="cursor-pointer font-mono tabular-nums">
                          {z.psi}{' '}
                          <span className="text-xs text-muted-foreground">{z.psi_ci}</span>
                        </span>
                      </TooltipTrigger>
                      <TooltipContent>
                        Sampling bounds [{z.psi_lo}–{z.psi_hi}] on a dataset-relative scale —
                        not a health threshold
                      </TooltipContent>
                    </Tooltip>
                  </TableCell>
                  <TableCell className="text-right">
                    <Badge
                      className={badgeTextClass(z.band)}
                      style={{ backgroundColor: bandColor }}
                    >
                      {z.band}
                    </Badge>
                  </TableCell>
                  <TableCell>
                    <div className="flex items-center gap-2">
                      <Progress value={z.cleanliness} className="h-2 w-20" />
                      <span className="font-mono text-xs tabular-nums">
                        {z.cleanliness}%
                      </span>
                    </div>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {z.total.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {z.clean.toLocaleString()}
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {z.dirty.toLocaleString()}{' '}
                    <span className="text-xs text-muted-foreground">
                      [{(z.dirty_rate_lo * 100).toFixed(1)}–{(z.dirty_rate_hi * 100).toFixed(1)}
                      %]
                    </span>
                  </TableCell>
                  <TableCell className="text-right font-mono tabular-nums">
                    {z.vote_agreement == null ? '—' : `${Math.round(z.vote_agreement * 100)}%`}
                  </TableCell>
                </TableRow>
              )
            })}
          </TableBody>
        </ShTable>
      </CardContent>
    </Card>
  )
}
