import { ChevronDown, Info } from 'lucide-react'
import {
  Collapsible,
  CollapsibleContent,
  CollapsibleTrigger,
} from '@/components/ui/collapsible'
import type { DashboardData } from '@/types'

interface Props {
  data: DashboardData | null
}

/** The honesty notes from the static dashboard, condensed and foldable. */
export function MethodologyNotes({ data }: Props) {
  if (!data) return null
  const weights = data.classes
    .map((c) => `${c} ×${data.hazard_weights[c]}`)
    .join(', ')

  return (
    <Collapsible className="rounded-xl border bg-card">
      <CollapsibleTrigger className="group flex w-full cursor-pointer items-center gap-2 px-4 py-3 text-left text-sm font-medium">
        <Info className="size-4 text-primary" aria-hidden />
        Methodology &amp; provenance
        <ChevronDown className="ml-auto size-4 text-muted-foreground transition-transform group-data-[state=open]:rotate-180" />
      </CollapsibleTrigger>
      <CollapsibleContent className="space-y-3 px-4 pb-4 text-xs leading-relaxed text-muted-foreground">
        <p>
          <b className="text-foreground">Cleanliness score</b> = clean count / total waste
          count × 100 (project definition; clean = recyclables, dirty = cigarette litter).
        </p>
        <p>
          <b className="text-foreground">PSI is a project-defined relative index, not a health
          index.</b> Sub-index<sub>c</sub> = count<sub>c</sub> × weight<sub>c</sub> ÷ max over
          all (zone, class) cells × 500; weights: {weights}. The 0–500 scale and band names are
          borrowed from air-quality indices (US EPA AQI / Singapore PSI / India CPCB NAQI) for
          readability only — values are an internal, dataset-relative ranking,{' '}
          <b>not validated</b> against any health outcome and <b>not comparable</b> to those
          indices or any environmental threshold.
        </p>
        <p>
          <b className="text-foreground">PSI ranges</b> are conditional sampling bounds: six
          Bonferroni-adjusted marginal Wilson intervals (nominal 95% joint level) mapped
          through the maximum weighted class index, holding the observed normalization
          reference fixed. They assume independent crops — they do not cover same-photo
          clustering, classifier error, or reference-estimation uncertainty. Bands use the
          point estimate.
        </p>
        <p>
          <b className="text-foreground">Predictions</b> are the majority vote of a{' '}
          {data.provenance.reps ?? '—'}-repeat completed experiment run ({data.provenance.run_id}
          , arm {data.provenance.arm}, classifier {data.provenance.clf}), read from
          manifest-verified artifacts whose SHA-256 hashes are checked on every request.
          "Vote agreement" is repeat-to-repeat prediction stability, not a calibrated
          probability of correctness.
        </p>
        <p>
          <b className="text-foreground">Zones.</b> TACO carries no GPS metadata; batch mode
          maps image batches to Zones A–F as an explicitly labelled demo. For a real
          deployment, place zone-labeled crops under <code>data/geo/&lt;Zone&gt;/&lt;class&gt;/</code>{' '}
          and switch to geo-folder mode — the analytics run unchanged.
        </p>
      </CollapsibleContent>
    </Collapsible>
  )
}
