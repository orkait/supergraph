import { memo } from 'react'
import { BaseEdge, getBezierPath, EdgeLabelRenderer, type EdgeProps } from '@xyflow/react'
import { useFlowStore } from '@/hooks/useFlowStore'
import { useSuperGraph } from '@/hooks/useSuperGraph'

const EDGE_COLORS: Record<string, string> = {
  calls: '#6a9fd8', extends: '#5aad7a', implements: '#9a80c8',
  imports: '#d4a84a', contains: '#5aaab8', uses: '#c878a0',
  calls_api: '#6a9fd8', reads_from: '#c87878',
  publishes_to: '#d4a84a', subscribes_to: '#8a7ab8',
}

function getEdgeColor(kind: string): string {
  if (EDGE_COLORS[kind]) return EDGE_COLORS[kind]
  const colors = Object.values(EDGE_COLORS)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export const CustomEdge = memo(function CustomEdge(props: EdgeProps) {
  const { sourceX, sourceY, targetX, targetY, source, target, data, style, id } = props
  const kind = (data?.kind as string) || ''
  const color = getEdgeColor(kind)

  // Primitive selectors - stable, no unnecessary re-renders
  const hoveredNodeId = useFlowStore((st) => st.hoveredNodeId)
  const highlightedEdges = useSuperGraph((st) => st.highlightedEdges)
  const highlightedNodeIds = useSuperGraph((st) => st.highlightedNodeIds)
  const showEdgeLabels = useSuperGraph((st) => st.config.showEdgeLabels)
  const layoutMode = useSuperGraph((st) => st.config.layoutMode)

  // Cluster mode: straight lines (freeform). Dagre mode: bezier curves.
  const [edgePath, labelX, labelY] = layoutMode === 'cluster'
    ? [`M ${sourceX},${sourceY} L ${targetX},${targetY}`, (sourceX + targetX) / 2, (sourceY + targetY) / 2] as [string, number, number]
    : getBezierPath({ sourceX, sourceY, targetX, targetY })

  const key = `${source}->${target}`
  const isHoverEdge = hoveredNodeId != null && (source === hoveredNodeId || target === hoveredNodeId)
  const isHoverDimmed = hoveredNodeId != null && !isHoverEdge
  const isQueryHighlighted = highlightedEdges.has(key)
  // Dim edge if: (1) edge-level highlights exist and this edge isn't one, OR
  //              (2) node-level highlights exist and neither endpoint is highlighted
  const hasAnyHighlights = highlightedEdges.size > 0 || highlightedNodeIds.size > 0
  const endpointHighlighted = highlightedNodeIds.has(source) && highlightedNodeIds.has(target)
  const isQueryDimmed = hasAnyHighlights && !isQueryHighlighted && !endpointHighlighted

  const highlighted = isHoverEdge || isQueryHighlighted
  const dimmed = isHoverDimmed || isQueryDimmed

  const isCrossGroup = kind === 'cross-group'
  const crossGroupCount = (data?.crossGroupCount as number) || 0
  const displayLabel = isCrossGroup
    ? `${crossGroupCount} edges`
    : kind

  const groupChildCount = (data?.groupChildCount as number) || 0
  const isGroupEdge = groupChildCount > 0
  const groupStrokeWidth = isGroupEdge ? Math.min(2 + groupChildCount * 0.05, 6) : undefined

  return (
    <>
      <BaseEdge
        path={edgePath}
        style={{
          ...style,
          stroke: highlighted ? 'var(--graph-highlight-border)' : dimmed ? 'var(--graph-edge-dimmed)' : color,
          strokeWidth: highlighted ? 3 : (groupStrokeWidth || 2),
          strokeDasharray: isCrossGroup ? '6 3' : undefined,
          opacity: dimmed ? 'var(--graph-edge-dimmed-opacity)' : 0.8,
          transition: 'opacity 0.2s, stroke 0.2s',
        }}
        id={id}
      />
      {showEdgeLabels && (
        <EdgeLabelRenderer>
          <div
            style={{
              position: 'absolute',
              transform: `translate(-50%, -50%) translate(${labelX}px, ${labelY}px)`,
              pointerEvents: 'none',
              fontSize: '10px',
              color: dimmed ? 'var(--graph-node-dimmed-text)' : 'var(--graph-edge-label-text)',
              backgroundColor: 'var(--graph-edge-label-bg)',
              padding: '1px 4px',
              borderRadius: '3px',
              opacity: dimmed ? 'var(--graph-edge-dimmed-opacity)' : 1,
              transition: 'opacity 0.2s',
            }}
          >
            {displayLabel}
          </div>
        </EdgeLabelRenderer>
      )}
    </>
  )
})
