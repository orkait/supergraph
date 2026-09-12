import { memo } from 'react'
import { Handle, Position, type NodeProps } from '@xyflow/react'
import { useFlowStore } from '@/hooks/useFlowStore'
import { useSuperGraph } from '@/hooks/useSuperGraph'

const NAMED_KINDS = new Set(['function', 'class', 'module', 'service', 'database', 'queue', 'default'])
const FALLBACK_NAMES = ['fallback-0', 'fallback-1', 'fallback-2']

function getKindTokenName(kind: string): string {
  if (NAMED_KINDS.has(kind)) return kind
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return FALLBACK_NAMES[Math.abs(hash) % FALLBACK_NAMES.length]
}

export const CustomNode = memo(function CustomNode({ id, data }: NodeProps) {
  const kind = (data.kind as string) || 'default'
  const degree = (data.degree as number) || 0
  const kindName = getKindTokenName(kind)

  // Read hover state - returns a primitive boolean, no new objects
  const isHoverTarget = useFlowStore((st) => st.hoveredNodeId === id)
  const isHoverNeighbor = useFlowStore((st) => {
    const hid = st.hoveredNodeId
    if (hid == null || hid === id) return false
    return st.edges.some(e => (e.source === hid && e.target === id) || (e.target === hid && e.source === id))
  })
  const isHoverActive = useFlowStore((st) => st.hoveredNodeId != null)

  const highlightedNodeIds = useSuperGraph((st) => st.highlightedNodeIds)
  const layoutMode = useSuperGraph((st) => st.config.layoutMode)

  const isHoverDimmed = isHoverActive && !isHoverTarget && !isHoverNeighbor
  const isQueryHighlighted = highlightedNodeIds.has(id) || Boolean(data.highlighted)
  const isQueryDimmed = highlightedNodeIds.size > 0 && !isQueryHighlighted

  const highlighted = isHoverTarget || isQueryHighlighted
  const dimmed = isHoverDimmed || isQueryDimmed

  const isGroup = Boolean(data.isGroup)
  const childCount = (data.childCount as number) || 0
  const groupParentId = data.groupParentId as string | undefined
  const groupKind = data.groupKind as string | undefined
  const toggleGroup = useSuperGraph((st) => st.toggleGroup)

  const scale = 1 + Math.min(degree * 0.08, 0.4)
  const minWidth = 140 + Math.min(degree * 8, 40)

  if (isGroup) {
    const handleClick = () => {
      if (groupParentId && groupKind) toggleGroup(groupParentId, groupKind)
    }
    return (
      <div
        onClick={handleClick}
        style={{
          backgroundColor: dimmed ? 'var(--graph-node-dimmed-bg)' : `var(--kind-${kindName}-bg)`,
          borderColor: highlighted
            ? 'var(--graph-highlight-border)'
            : dimmed ? 'var(--graph-node-dimmed-border)'
            : `var(--kind-${kindName}-border)`,
          borderWidth: highlighted ? '2px' : '1px',
          borderStyle: 'dashed',
          opacity: dimmed ? 'var(--graph-dimmed-opacity)' : 1,
          transition: 'opacity 0.2s, background-color 0.2s, border-color 0.2s',
          minWidth: '180px',
          cursor: 'pointer',
        }}
        className="px-3 py-2 rounded-lg border"
      >
        <Handle type="target" position={Position.Top} className="!w-2 !h-2" style={{ background: `var(--kind-${kindName}-border)` }} />
        <div
          className="text-xs font-semibold truncate"
          style={{ color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-text)` }}
        >
          {kind} ({childCount})
        </div>
        <div className="flex items-center gap-1.5 mt-1">
          <div
            className="text-[9px] inline-block px-1.5 py-0.5 rounded"
            style={{
              backgroundColor: `color-mix(in srgb, var(--kind-${kindName}-badge) 13%, transparent)`,
              color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-badge)`,
              border: dimmed
                ? '1px solid var(--graph-node-dimmed-text)'
                : `1px solid color-mix(in srgb, var(--kind-${kindName}-badge) 27%, transparent)`,
            }}
          >
            click to expand
          </div>
          {data.matchCount != null && (data.matchCount as number) > 0 && (
            <div className="text-[8px] px-1 py-0.5 rounded" style={{ color: 'var(--graph-highlight-border)', backgroundColor: 'var(--graph-degree-bg)' }}>
              {data.matchCount as number} matched
            </div>
          )}
        </div>
        <Handle type="source" position={Position.Bottom} className="!w-2 !h-2" style={{ background: `var(--kind-${kindName}-border)` }} />
      </div>
    )
  }

  // Cluster mode: circular Obsidian-style nodes
  if (layoutMode === 'cluster') {
    const label = data.label as string
    // Size based on label length + degree - ensure text fits
    const baseSize = Math.max(50, label.length * 6 + 20)
    const size = baseSize + Math.min(degree * 4, 24)
    return (
      <div
        style={{
          width: `${size}px`,
          height: `${size}px`,
          borderRadius: '50%',
          backgroundColor: dimmed ? 'var(--graph-node-dimmed-bg)' : `var(--kind-${kindName}-bg)`,
          borderColor: highlighted
            ? 'var(--graph-highlight-border)'
            : dimmed ? 'var(--graph-node-dimmed-border)'
            : `var(--kind-${kindName}-border)`,
          borderWidth: highlighted ? '2px' : '1px',
          borderStyle: 'solid',
          boxShadow: highlighted
            ? '0 0 20px var(--graph-highlight-shadow)'
            : dimmed ? 'none'
            : `0 0 ${8 + degree * 2}px color-mix(in srgb, var(--kind-${kindName}-border) 40%, transparent)`,
          opacity: dimmed ? 'var(--graph-dimmed-opacity)' : 1,
          transition: 'opacity 0.15s, box-shadow 0.15s',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          cursor: 'grab',
          position: 'relative',
        }}
      >
        <Handle type="target" position={Position.Top} className="!w-0 !h-0 !min-w-0 !min-h-0 !border-0 !bg-transparent" style={{ left: '50%', top: '50%' }} />
        <div
          className="text-[10px] font-semibold text-center leading-tight"
          style={{
            color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-text)`,
            maxWidth: `${size - 12}px`,
            wordBreak: 'break-word',
            padding: '4px',
          }}
        >
          {label}
        </div>
        <Handle type="source" position={Position.Bottom} className="!w-0 !h-0 !min-w-0 !min-h-0 !border-0 !bg-transparent" style={{ left: '50%', top: '50%' }} />
      </div>
    )
  }

  // Dagre mode: rectangular card nodes
  return (
    <div
      style={{
        backgroundColor: dimmed ? 'var(--graph-node-dimmed-bg)' : `var(--kind-${kindName}-bg)`,
        borderColor: highlighted
          ? 'var(--graph-highlight-border)'
          : dimmed ? 'var(--graph-node-dimmed-border)'
          : `var(--kind-${kindName}-border)`,
        borderWidth: highlighted ? '2px' : '1px',
        boxShadow: highlighted
          ? '0 0 16px var(--graph-highlight-shadow)'
          : dimmed ? 'none'
          : `0 0 ${6 + degree * 2}px color-mix(in srgb, var(--kind-${kindName}-border) 27%, transparent)`,
        opacity: dimmed ? 'var(--graph-dimmed-opacity)' : 1,
        transition: 'opacity 0.2s, background-color 0.2s, border-color 0.2s, box-shadow 0.2s',
        transform: `scale(${dimmed ? 0.95 : scale})`,
        minWidth: `${minWidth}px`,
      }}
      className="px-3 py-2 rounded-lg border"
    >
      <Handle type="target" position={Position.Top} className="!w-2 !h-2" style={{ background: `var(--kind-${kindName}-border)` }} />
      <div
        className="text-xs font-semibold truncate"
        style={{ color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-text)`, fontSize: `${11 + Math.min(degree, 3)}px` }}
      >
        {data.label as string}
      </div>
      <div className="flex items-center gap-1.5 mt-1">
        <div
          className="text-[9px] inline-block px-1.5 py-0.5 rounded"
          style={{
            backgroundColor: `color-mix(in srgb, var(--kind-${kindName}-badge) 13%, transparent)`,
            color: dimmed ? 'var(--graph-node-dimmed-text)' : `var(--kind-${kindName}-badge)`,
            border: dimmed
              ? '1px solid var(--graph-node-dimmed-text)'
              : `1px solid color-mix(in srgb, var(--kind-${kindName}-badge) 27%, transparent)`,
          }}
        >
          {kind}
        </div>
        {degree > 0 && !dimmed && (
          <div className="text-[8px] px-1 py-0.5 rounded" style={{ color: 'var(--graph-degree-text)', backgroundColor: 'var(--graph-degree-bg)' }}>
            {degree}
          </div>
        )}
      </div>
      <Handle type="source" position={Position.Bottom} className="!w-2 !h-2" style={{ background: `var(--kind-${kindName}-border)` }} />
    </div>
  )
})
