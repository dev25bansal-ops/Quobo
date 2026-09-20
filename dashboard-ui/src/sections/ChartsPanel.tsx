import { Bar, BarChart, CartesianGrid, LabelList, XAxis, YAxis } from 'recharts'
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from '@/components/ui/card'
import { Skeleton } from '@/components/ui/skeleton'
import {
  type ChartConfig,
  ChartContainer,
  ChartTooltip,
  ChartTooltipContent,
} from '@/components/ui/chart'
import { classColor } from '@/lib/colors'
import type { DashboardData } from '@/types'

interface Props {
  data: DashboardData | null
  loading: boolean
  selectedZone: string | null
  onSelectZone: (z: string | null) => void
}

const psiConfig: ChartConfig = {
  psi: { label: 'Relative PSI' },
}

export function ChartsPanel({ data, loading, selectedZone, onSelectZone }: Props) {
  if (loading && !data) {
    return (
      <div className="grid gap-4 lg:grid-cols-2">
        <Skeleton className="h-[340px] rounded-xl" />
        <Skeleton className="h-[340px] rounded-xl" />
      </div>
    )
  }
  if (!data || data.zones.length === 0) return null

  const psiData = data.zones.map((z) => ({
    zone: z.zone,
    psi: z.psi,
    band: z.band,
    ci: z.psi_ci,
    fill: data.band_colors[z.band] ?? '#475569',
  }))

  const compData = data.zones.map((z) => {
    const row: Record<string, string | number> = { zone: z.zone }
    for (const c of data.classes) {
      const n = z.counts[c] ?? 0
      row[c] = z.total ? Math.round((n / z.total) * 1000) / 10 : 0
      row[`_${c}`] = n
    }
    return row
  })

  const compConfig: ChartConfig = Object.fromEntries(
    data.classes.map((c, i) => [
      c,
      { label: `${c} (×${data.hazard_weights[c] ?? 1} weight)`, color: classColor(c, i) },
    ]),
  )

  return (
    <div className="grid gap-4 lg:grid-cols-2">
      <Card>
        <CardHeader>
          <CardTitle>Relative PSI by zone</CardTitle>
          <CardDescription>
            Hazard-weighted dominant-class index, rescaled 0–500 against the worst observed
            (zone, class) cell — dataset-relative, not a health measure. Hover for sampling
            bounds.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={psiConfig} className="h-[260px] w-full">
            <BarChart data={psiData} margin={{ top: 8, right: 8, left: -16 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="zone" tickLine={false} axisLine={false} fontSize={12} />
              <YAxis domain={[0, 500]} tickLine={false} axisLine={false} fontSize={12} />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    formatter={(value, _name, item) => (
                      <>
                        <div className="flex items-center gap-1.5 font-mono text-xs tabular-nums">
                          <span
                            className="h-2 w-2 shrink-0 rounded-sm"
                            style={{ backgroundColor: String(item?.payload?.fill) }}
                          />
                          PSI {String(value)}
                          <span className="text-muted-foreground">
                            ({String(item?.payload?.ci)}) · {String(item?.payload?.band)}
                          </span>
                        </div>
                      </>
                    )}
                  />
                }
              />
              <Bar
                dataKey="psi"
                radius={[4, 4, 0, 0]}
                className="cursor-pointer transition-opacity hover:opacity-80"
                onClick={(d) => onSelectZone(selectedZone === d.zone ? null : d.zone)}
              >
                <LabelList
                  dataKey="psi"
                  position="top"
                  fontSize={11}
                  formatter={(v: number) => String(v)}
                />
              </Bar>
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>

      <Card>
        <CardHeader>
          <CardTitle>Class composition by zone</CardTitle>
          <CardDescription>
            Share of each waste class per zone (100% stacked). Hover for counts; legend shows
            hazard weights.
          </CardDescription>
        </CardHeader>
        <CardContent>
          <ChartContainer config={compConfig} className="h-[260px] w-full">
            <BarChart data={compData} margin={{ top: 8, right: 8, left: -16 }}>
              <CartesianGrid vertical={false} strokeDasharray="3 3" />
              <XAxis dataKey="zone" tickLine={false} axisLine={false} fontSize={12} />
              <YAxis domain={[0, 100]} tickLine={false} axisLine={false} fontSize={12} unit="%" />
              <ChartTooltip
                cursor={false}
                content={
                  <ChartTooltipContent
                    formatter={(value, name, item) => (
                      <div className="flex items-center gap-1.5 font-mono text-xs tabular-nums">
                        <span
                          className="h-2 w-2 shrink-0 rounded-sm"
                          style={{
                            backgroundColor: (item?.payload as Record<string, unknown>)
                              ? `var(--color-${String(name)})`
                              : undefined,
                          }}
                        />
                        {String(name)}: {String(value)}%
                        <span className="text-muted-foreground">
                          ({String((item?.payload as Record<string, unknown>)?.[`_${String(name)}`])}{' '}
                          items)
                        </span>
                      </div>
                    )}
                  />
                }
              />
              {data.classes.map((c) => (
                <Bar key={c} dataKey={c} stackId="composition" fill={`var(--color-${c})`} />
              ))}
            </BarChart>
          </ChartContainer>
        </CardContent>
      </Card>
    </div>
  )
}
