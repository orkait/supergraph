import {
  ReactFlow,
  Controls,
  MiniMap,
  Background,
  BackgroundVariant,
  type Node,
  type Edge,
  type NodeMouseHandler,
} from '@xyflow/react'
import '@xyflow/react/dist/style.css'
import { CustomNode } from '@/components/graph/CustomNode'
import { CustomEdge } from '@/components/graph/CustomEdge'
import { CanvasControls } from '@/components/graph/CanvasControls'
import { useSuperGraph } from '@/hooks/useSuperGraph'
import { useFlowStore } from '@/hooks/useFlowStore'
import { applyDagreLayout } from '@/components/graph/layout'
import { collapseTransform, type GroupNodeData } from '@/components/graph/collapseTransform'
import { createLiveSimulation, updateSimulationForces } from '@/components/graph/forceLayout'
import type { Simulation } from 'd3-force'
import { Card, CardContent, CardHeader } from '@/components/ui/card'
import { useMemo, useEffect, useRef, useState, useCallback } from 'react'
import { Search, Loader2 } from 'lucide-react'

const nodeTypes = { custom: CustomNode }
const edgeTypes = { custom: CustomEdge }

function computeDegrees(edges: { source: string; target: string }[]) {
  const deg: Record<string, number> = {}
  for (const e of edges) {
    deg[e.source] = (deg[e.source] || 0) + 1
    deg[e.target] = (deg[e.target] || 0) + 1
  }
  return deg
}

function getEdgeColor(kind: string): string {
  const map: Record<string, string> = {
    calls: '#6a9fd8', extends: '#5aad7a', implements: '#9a80c8',
    imports: '#d4a84a', contains: '#5aaab8', uses: '#c878a0',
    calls_api: '#6a9fd8', reads_from: '#c87878',
    publishes_to: '#d4a84a', subscribes_to: '#8a7ab8',
  }
  if (map[kind]) return map[kind]
  const colors = Object.values(map)
  let hash = 0
  for (let i = 0; i < kind.length; i++) hash = kind.charCodeAt(i) + ((hash << 5) - hash)
  return colors[Math.abs(hash) % colors.length]
}

export function GraphPanel() {
  const graph = useSuperGraph((s) => s.graph)
  const config = useSuperGraph((s) => s.config)
  const { showMinimap, isDark, nodesep, ranksep, layoutDirection } = config
  const { collapseThreshold } = config
  const expandedGroups = useSuperGraph((s) => s.expandedGroups)
  const highlightedNodeIds = useSuperGraph((s) => s.highlightedNodeIds)
  const highlightedEdges = useSuperGraph((s) => s.highlightedEdges)
  const results = useSuperGraph((s) => s.results)
  const lastResultKind = results.length > 0 ? results[0]?.result?.kind : undefined
  const isPathResult = lastResultKind === 'path' || lastResultKind === 'paths'
  const loading = useSuperGraph((s) => s.loading)

  const nodes = useFlowStore((s) => s.nodes)
  const edges = useFlowStore((s) => s.edges)
  const onNodesChange = useFlowStore((s) => s.onNodesChange)
  const onEdgesChange = useFlowStore((s) => s.onEdgesChange)
  const setFlowNodes = useFlowStore((s) => s.setNodes)
  const setFlowEdges = useFlowStore((s) => s.setEdges)
  const setHoveredNodeId = useFlowStore((s) => s.setHoveredNodeId)

  const [searchFilter, setSearchFilter] = useState('')
  const prevStructureRef = useRef('')
  const proOptions = useMemo(() => ({ hideAttribution: true }), [])
  // eslint-disable-next-line @typescript-eslint/no-explicit-any
  const simulationRef = useRef<{ simulation: Simulation<any, any>; nodeMap: Map<string, any> } | null>(null)
  const containerRef = useRef<HTMLDivElement>(null)

  const degrees = useMemo(() => computeDegrees(graph.edges), [graph.edges])

  // Single effect: layout ONLY when graph structure or search changes
  // Highlight/dim is computed inside CustomNode/CustomEdge directly
  useEffect(() => {
    const structureKey = JSON.stringify({
      n: graph.nodes.map(n => n.id).sort(),
      e: graph.edges.map(e => `${e.source}->${e.target}`).sort(),
      f: searchFilter,
      lm: config.layoutMode,
      h: [...highlightedNodeIds].sort(),
      he: [...highlightedEdges].sort(),
      // Dagre-specific layout params
      ...(config.layoutMode === 'dagre' ? {
        ns: nodesep, rs: ranksep, ld: layoutDirection, ct: collapseThreshold, eg: expandedGroups,
      } : {}),
    })

    if (structureKey === prevStructureRef.current) return
    prevStructureRef.current = structureKey

    const filteredGraphNodes = searchFilter
      ? graph.nodes.filter(n =>
          n.id.toLowerCase().includes(searchFilter.toLowerCase()) ||
          (n.name || '').toLowerCase().includes(searchFilter.toLowerCase()) ||
          (n.kind || '').toLowerCase().includes(searchFilter.toLowerCase())
        )
      : graph.nodes
    const filteredIds = new Set(filteredGraphNodes.map(n => n.id))

    // For path results, force-expand all groups along the path
    let effectiveExpanded = expandedGroups
    if (isPathResult && highlightedNodeIds.size > 0 && config.layoutMode === 'dagre') {
      effectiveExpanded = { ...expandedGroups }
      const childrenMap = new Map<string, string[]>()
      for (const e of graph.edges) {
        if (!filteredIds.has(e.source) || !filteredIds.has(e.target)) continue
        const list = childrenMap.get(e.source) || []
        list.push(e.target)
        childrenMap.set(e.source, list)
      }
      const nodeMap = new Map(filteredGraphNodes.map(n => [n.id, n]))
      for (const [parentId, children] of childrenMap) {
        if (children.length <= collapseThreshold) continue
        for (const childId of children) {
          if (highlightedNodeIds.has(childId)) {
            const childNode = nodeMap.get(childId)
            const kind = childNode?.kind || 'default'
            const list = effectiveExpanded[parentId] || []
            if (!list.includes(kind)) {
              effectiveExpanded[parentId] = [...list, kind]
            }
          }
        }
      }
    }

    // Apply collapse transform for dagre mode
    const { nodes: collapsedNodes, edges: collapsedEdges, groupMeta } =
      config.layoutMode === 'dagre'
        ? collapseTransform(
            filteredGraphNodes,
            graph.edges.filter(e => filteredIds.has(e.source) && filteredIds.has(e.target)),
            collapseThreshold,
            effectiveExpanded,
            highlightedNodeIds.size > 0 ? highlightedNodeIds : undefined
          )
        : { nodes: filteredGraphNodes, edges: graph.edges.filter(e => filteredIds.has(e.source) && filteredIds.has(e.target)), groupMeta: new Map<string, GroupNodeData>() }
    const collapsedIds = new Set(collapsedNodes.map(n => n.id))

    const rfNodes: Node[] = collapsedNodes.map((n) => {
      const meta = groupMeta.get(n.id)
      return {
        id: n.id,
        type: 'custom',
        position: { x: 0, y: 0 },
        data: meta
          ? {
              label: n.id,
              kind: meta.kind,
              degree: meta.childCount,
              fields: n,
              isGroup: true,
              childCount: meta.childCount,
              groupParentId: meta.parentId,
              groupKind: meta.kind,
              matchCount: meta.matchCount,
              highlighted: meta.matchCount > 0,
            }
          : {
              label: n.id,
              kind: n.kind,
              degree: degrees[n.id] || 0,
              fields: n,
              highlighted: highlightedNodeIds.has(n.id),
            },
      }
    })

    const rfEdges: Edge[] = collapsedEdges
      .filter(e => collapsedIds.has(e.source) && collapsedIds.has(e.target))
      .map((e, i) => ({
        id: `e-${i}-${e.source}-${e.target}-${e.kind}`,
        source: e.source,
        target: e.target,
        type: 'custom',
        data: {
          kind: e.kind,
          crossGroupCount: e.crossGroupCount,
          groupChildCount: e.groupChildCount,
        },
      }))

    if (config.layoutMode === 'cluster') {
      // Capture existing positions before stopping simulation (for smooth highlight transitions)
      const existingPositions = new Map<string, { x: number; y: number }>()
      if (simulationRef.current) {
        for (const [id, fn] of simulationRef.current.nodeMap) {
          if (fn.x != null && fn.y != null) {
            existingPositions.set(id, { x: fn.x, y: fn.y })
          }
        }
      }

      // Stop any previous simulation
      simulationRef.current?.simulation.stop()
      simulationRef.current = null

      const layoutNodes = rfNodes
      const layoutEdges = rfEdges

      const container = containerRef.current
      const width = container?.clientWidth || 800
      const height = container?.clientHeight || 600

      // Set edges immediately
      setFlowEdges(layoutEdges)

      // Create a LIVE simulation - preserve positions if available
      const live = createLiveSimulation(layoutNodes, layoutEdges, {
        clusterStrength: config.clusterStrength,
        repelStrength: config.repelStrength,
        centerForce: config.centerForce,
        linkForce: config.linkForce,
        linkDistance: config.linkDistance,
        width,
        height,
      }, (updatedNodes) => {
        setFlowNodes(updatedNodes)
      }, existingPositions.size > 0 ? existingPositions : undefined)
      simulationRef.current = live
    } else {
      // Dagre layout
      const layoutOpts = { direction: layoutDirection, nodesep, ranksep }
      setFlowNodes(applyDagreLayout(rfNodes, rfEdges, layoutOpts))
      setFlowEdges(rfEdges)
    }
  }, [graph, searchFilter, degrees, highlightedNodeIds, highlightedEdges, nodesep, ranksep, layoutDirection, expandedGroups, collapseThreshold, config.layoutMode, config.clusterStrength, config.repelStrength, config.centerForce, config.linkForce, config.linkDistance, setFlowNodes, setFlowEdges, results, isPathResult])

  // Cleanup simulation and force fresh layout when mode changes
  useEffect(() => {
    // Reset structureKey so the layout effect re-runs from scratch
    prevStructureRef.current = ''
    return () => {
      simulationRef.current?.simulation.stop()
      simulationRef.current = null
    }
  }, [config.layoutMode])

  // Update forces live when sliders change (without resetting the simulation)
  useEffect(() => {
    if (config.layoutMode !== 'cluster' || !simulationRef.current) return
    const container = containerRef.current
    updateSimulationForces(simulationRef.current.simulation, {
      clusterStrength: config.clusterStrength,
      repelStrength: config.repelStrength,
      centerForce: config.centerForce,
      linkForce: config.linkForce,
      linkDistance: config.linkDistance,
      width: container?.clientWidth || 800,
      height: container?.clientHeight || 600,
    })
  }, [config.clusterStrength, config.repelStrength, config.centerForce, config.linkForce, config.linkDistance, config.layoutMode])

  // --- Cluster mode drag: simple pin/unpin ---
  const onNodeDragStart = useCallback((_event: React.MouseEvent, node: Node) => {
    if (config.layoutMode !== 'cluster' || !simulationRef.current) return
    const { simulation, nodeMap } = simulationRef.current
    const fn = nodeMap.get(node.id)
    if (!fn) return
    fn.fx = node.position.x
    fn.fy = node.position.y
    simulation.alphaTarget(0.3).restart()
  }, [config.layoutMode])

  const onNodeDrag = useCallback((_event: React.MouseEvent, node: Node) => {
    if (config.layoutMode !== 'cluster' || !simulationRef.current) return
    const fn = simulationRef.current.nodeMap.get(node.id)
    if (!fn) return
    fn.fx = node.position.x
    fn.fy = node.position.y
  }, [config.layoutMode])

  const onNodeDragStop = useCallback((_event: React.MouseEvent, node: Node) => {
    if (config.layoutMode !== 'cluster' || !simulationRef.current) return
    const { simulation, nodeMap } = simulationRef.current
    const fn = nodeMap.get(node.id)
    if (!fn) return
    fn.fx = null
    fn.fy = null
    simulation.alphaTarget(0.02)
  }, [config.layoutMode])

  const onNodeMouseEnter: NodeMouseHandler = useCallback((_event, node) => {
    setHoveredNodeId(node.id)
  }, [setHoveredNodeId])

  const onNodeMouseLeave: NodeMouseHandler = useCallback(() => {
    setHoveredNodeId(null)
  }, [setHoveredNodeId])

  return (
    <Card className="h-full gap-0 py-0 rounded-lg flex flex-col">
      <CardHeader className="px-3 py-2 border-b flex flex-row items-center justify-between space-y-0 h-[40px] flex-shrink-0">
        <div className="text-xs text-muted-foreground flex items-center gap-2 flex-1">
          <Search className="w-3 h-3" />
          <input
            type="text"
            placeholder="Filter nodes..."
            value={searchFilter}
            onChange={(e) => setSearchFilter(e.target.value)}
            className="flex-1 bg-transparent text-xs text-foreground placeholder:text-muted-foreground outline-none font-normal"
          />
          {searchFilter && (
            <button onClick={() => setSearchFilter('')} className="text-[10px] text-muted-foreground hover:text-foreground">
              Clear
            </button>
          )}
        </div>
        <div className="flex items-center gap-1.5 ml-4">
          {loading && <Loader2 className="w-3.5 h-3.5 animate-spin text-muted-foreground" />}
        </div>
      </CardHeader>
      <CardContent ref={containerRef} className="flex-1 min-h-0 overflow-hidden p-0 relative">
        {searchFilter && nodes.length === 0 && (
          <div className="absolute inset-0 flex items-center justify-center z-10 pointer-events-none">
            <div className="text-center text-muted-foreground/60">
              <Search className="w-8 h-8 mx-auto mb-2 opacity-50" />
              <p className="text-sm">No nodes match &lsquo;{searchFilter}&rsquo;</p>
            </div>
          </div>
        )}
        <ReactFlow
          nodes={nodes}
          edges={edges}
          onNodesChange={onNodesChange}
          onEdgesChange={onEdgesChange}
          onNodeMouseEnter={onNodeMouseEnter}
          onNodeMouseLeave={onNodeMouseLeave}
          onNodeDragStart={onNodeDragStart}
          onNodeDrag={onNodeDrag}
          onNodeDragStop={onNodeDragStop}
          nodeTypes={nodeTypes}
          edgeTypes={edgeTypes}
          proOptions={proOptions}
          fitView
          nodesDraggable
          onlyRenderVisibleElements
          elevateNodesOnSelect={false}
          elevateEdgesOnSelect={false}
          className="bg-background"
        >
          <Background variant={BackgroundVariant.Dots} gap={20} size={1} color="var(--graph-bg-dots)" />
          <Controls className="!bg-card !border-border !text-foreground [&>button]:!bg-card [&>button]:!border-border [&>button]:!text-foreground [&>button:hover]:!bg-accent" />
          <CanvasControls />
          {showMinimap && (
            <MiniMap
              className="!bg-card !border-border"
              nodeColor={(n) => getEdgeColor((n.data?.kind as string) || '')}
              maskColor={isDark ? 'rgba(0,0,0,0.7)' : 'rgba(255,255,255,0.7)'}
            />
          )}
        </ReactFlow>
      </CardContent>
    </Card>
  )
}
