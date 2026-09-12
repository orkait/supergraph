import { Panel } from '@xyflow/react'
import { useSuperGraph, type LayoutMode } from '@/hooks/useSuperGraph'
import { Slider } from '@/components/ui/slider'

export function CanvasControls() {
  const config = useSuperGraph((s) => s.config)
  const updateConfig = useSuperGraph((s) => s.updateConfig)
  const { layoutMode, clusterStrength, repelStrength, centerForce, linkForce, linkDistance } = config

  return (
    <Panel position="top-right">
      <div className="bg-[var(--panel-bg)] backdrop-blur-sm border border-[var(--panel-border)] rounded-lg p-3 space-y-3 min-w-[180px]">
        {/* Mode toggle */}
        <div className="flex rounded-md overflow-hidden border border-[var(--panel-border)] text-[11px]">
          <button
            onClick={() => updateConfig({ layoutMode: 'dagre' as LayoutMode })}
            className={`flex-1 px-3 py-1.5 transition-colors ${
              layoutMode === 'dagre'
                ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)] border-r border-[var(--btn-primary-border)]'
                : 'bg-[var(--panel-bg)] text-[var(--panel-text)] hover:bg-[var(--panel-hover)] border-r border-[var(--panel-border)]'
            }`}
          >
            Dagre
          </button>
          <button
            onClick={() => updateConfig({ layoutMode: 'cluster' as LayoutMode })}
            className={`flex-1 px-3 py-1.5 transition-colors ${
              layoutMode === 'cluster'
                ? 'bg-[var(--btn-primary-bg)] text-[var(--btn-primary-text)]'
                : 'bg-[var(--panel-bg)] text-[var(--panel-text)] hover:bg-[var(--panel-hover)]'
            }`}
          >
            Cluster
          </button>
        </div>

        {/* Cluster-specific: all sliders 0-100 */}
        {layoutMode === 'cluster' && (
          <>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-[var(--panel-label)]">Center force</span>
                <span className="text-[10px] font-mono text-[var(--panel-value)]">{centerForce}</span>
              </div>
              <Slider
                value={[centerForce]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ centerForce: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-[var(--panel-label)]">Repel force</span>
                <span className="text-[10px] font-mono text-[var(--panel-value)]">{repelStrength}</span>
              </div>
              <Slider
                value={[repelStrength]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ repelStrength: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-[var(--panel-label)]">Link force</span>
                <span className="text-[10px] font-mono text-[var(--panel-value)]">{linkForce}</span>
              </div>
              <Slider
                value={[linkForce]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ linkForce: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-[var(--panel-label)]">Link distance</span>
                <span className="text-[10px] font-mono text-[var(--panel-value)]">{linkDistance}</span>
              </div>
              <Slider
                value={[linkDistance]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ linkDistance: Array.isArray(v) ? v[0] : v })}
              />
            </div>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between">
                <span className="text-[10px] text-[var(--panel-label)]">Cluster force</span>
                <span className="text-[10px] font-mono text-[var(--panel-value)]">{clusterStrength}</span>
              </div>
              <Slider
                value={[clusterStrength]}
                min={0} max={100} step={1}
                onValueChange={(v) => updateConfig({ clusterStrength: Array.isArray(v) ? v[0] : v })}
              />
            </div>
          </>
        )}
      </div>
    </Panel>
  )
}
