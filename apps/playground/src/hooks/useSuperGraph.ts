import { create } from 'zustand'
import { persist } from 'zustand/middleware'
import { toast } from 'sonner'
import { api, type Result, type GraphData } from '@/api/client'
import { classHierarchy } from '@/examples/class-hierarchy'

export type LayoutMode = 'dagre' | 'cluster'
export type LayoutDirection = 'TB' | 'LR'

export interface ResultEntry {
  id: string
  query: string
  result: Result | null
  error: string | null
  timestamp: number
}

export interface Config {
  layoutMode: LayoutMode
  showEdgeLabels: boolean
  showMinimap: boolean
  colorByKind: boolean
  ceilingMb: number
  costThreshold: number
  showElapsed: boolean
  isDark: boolean  // client-only; not synced to server
  // Layout tuning (client-only)
  nodesep: number
  ranksep: number
  layoutDirection: LayoutDirection
  fontSize: number  // editor font size (client-only)
  collapseThreshold: number  // dagre mode: auto-collapse above this child count
  clusterStrength: number    // cluster mode: 0-100 normalized
  repelStrength: number      // cluster mode: 0-100 normalized
  centerForce: number        // cluster mode: 0-100 normalized
  linkForce: number          // cluster mode: 0-100 normalized
  linkDistance: number        // cluster mode: 0-100 normalized
}

interface SuperGraphState {
  graph: GraphData
  results: ResultEntry[]
  config: Config
  editorContent: string
  editorSelection: string
  highlightedNodeIds: Set<string>
  highlightedEdges: Set<string>
  activeResultId: string | null
  loading: boolean
  expandedGroups: Record<string, string[]>

  setEditorContent: (content: string) => void
  setEditorSelection: (selection: string) => void
  executeQuery: (query: string, skipRefresh?: boolean) => Promise<void>
  executeSelected: () => Promise<void>
  executeAll: () => Promise<void>
  refreshGraph: () => Promise<void>
  resetGraph: () => Promise<void>
  clearResults: () => void
  selectResult: (id: string) => void
  clearHighlights: () => void
  updateConfig: (partial: Partial<Config>) => void
  toggleGroup: (parentId: string, kind: string) => void
  resetExpandedGroups: () => void
}

let resultCounter = 0

function splitQueries(text: string): string[] {
  const lines = text.split('\n')
  const queries: string[] = []
  let batch: string[] = []
  let inBatch = false

  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('//')) continue

    if (trimmed === 'BEGIN') {
      inBatch = true
      batch = [line]
    } else if (trimmed === 'COMMIT' && inBatch) {
      batch.push(line)
      queries.push(batch.join('\n'))
      batch = []
      inBatch = false
    } else if (inBatch) {
      batch.push(line)
    } else {
      queries.push(trimmed)
    }
  }
  return queries
}

function extractHighlights(result: Result): { nodeIds: Set<string>; edgeKeys: Set<string> } {
  const nodeIds = new Set<string>()
  const edgeKeys = new Set<string>()
  if (!result.data) return { nodeIds, edgeKeys }

  switch (result.kind) {
    case 'node':
      if (result.data?.id) nodeIds.add(result.data.id)
      break
    case 'nodes':
      if (Array.isArray(result.data))
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        result.data.forEach((n: any) => n?.id && nodeIds.add(n.id))
      break
    case 'edges':
      if (Array.isArray(result.data))
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        result.data.forEach((e: any) => {
          if (e.source) nodeIds.add(e.source)
          if (e.target) nodeIds.add(e.target)
          edgeKeys.add(`${e.source}->${e.target}`)
        })
      break
    case 'path':
      if (Array.isArray(result.data)) {
        result.data.forEach((id: string) => nodeIds.add(id))
        for (let i = 0; i < result.data.length - 1; i++)
          edgeKeys.add(`${result.data[i]}->${result.data[i + 1]}`)
      }
      break
    case 'paths':
      if (Array.isArray(result.data))
        result.data.forEach((path: string[]) => {
          path.forEach((id: string) => nodeIds.add(id))
          for (let i = 0; i < path.length - 1; i++)
            edgeKeys.add(`${path[i]}->${path[i + 1]}`)
        })
      break
    case 'subgraph':
      if (result.data?.nodes)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        result.data.nodes.forEach((n: any) => n?.id && nodeIds.add(n.id))
      if (result.data?.edges)
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        result.data.edges.forEach((e: any) => edgeKeys.add(`${e.source}->${e.target}`))
      break
    case 'match':
      if (Array.isArray(result.data?.bindings))
        result.data.bindings.forEach((binding: Record<string, string>) =>
          Object.values(binding).forEach(id => nodeIds.add(id)))
      if (Array.isArray(result.data?.edges))
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        result.data.edges.forEach((e: any) => {
          if (e.source) nodeIds.add(e.source)
          if (e.target) nodeIds.add(e.target)
          edgeKeys.add(`${e.source}->${e.target}`)
        })
      break
  }
  return { nodeIds, edgeKeys }
}

export const useSuperGraph = create<SuperGraphState>()(
  persist(
    (set, get) => ({
      graph: { nodes: [], edges: [] },
      results: [],
      config: {
        layoutMode: 'dagre',
        showEdgeLabels: true,
        showMinimap: true,
        colorByKind: true,
        ceilingMb: 256,
        costThreshold: 100_000,
        showElapsed: true,
        isDark: true,
        nodesep: 50,
        ranksep: 80,
        layoutDirection: 'TB',
        fontSize: 14,
        collapseThreshold: 20,
        clusterStrength: 50,
        repelStrength: 33,
        centerForce: 11,
        linkForce: 25,
        linkDistance: 32,
      },
      editorContent: classHierarchy.script,
      editorSelection: '',
      highlightedNodeIds: new Set(),
      highlightedEdges: new Set(),
      activeResultId: null,
      loading: false,
      expandedGroups: {},

      setEditorContent: (content) => set({ editorContent: content }),
      setEditorSelection: (selection) => set({ editorSelection: selection }),

      executeSelected: async () => {
        const sel = get().editorSelection
        if (sel.trim()) {
          const queries = splitQueries(sel)
          for (const q of queries) await get().executeQuery(q, true)
          await get().refreshGraph()
        }
      },

      executeQuery: async (query, skipRefresh = false) => {
        set({ loading: true })
        const id = `r-${++resultCounter}`
        try {
          const result = await api.execute(query)
          if (result.kind === 'error') {
            set((s) => ({
              results: [{ id, query, result: null, error: result.data as string, timestamp: Date.now() }, ...s.results].slice(0, 50),
              loading: false
            }))
          } else {
            const { nodeIds, edgeKeys } = extractHighlights(result)
            const hasHighlights = nodeIds.size > 0 || edgeKeys.size > 0
            set((s) => ({
              results: [{ id, query, result, error: null, timestamp: Date.now() }, ...s.results].slice(0, 50),
              // Only update highlights if the result produces them
              ...(hasHighlights ? {
                highlightedNodeIds: nodeIds,
                highlightedEdges: edgeKeys,
                activeResultId: id,
              } : {}),
              loading: false
            }))
            if (result.kind === 'ok' && !skipRefresh) await get().refreshGraph()
          }
        // eslint-disable-next-line @typescript-eslint/no-explicit-any
        } catch (err: any) {
          const msg = err?.error || String(err)
          toast.error('Query failed', { description: msg })
          set((s) => ({
            results: [{ id, query, result: null, error: msg, timestamp: Date.now() }, ...s.results].slice(0, 50),
            loading: false
          }))
        }
      },

      executeAll: async () => {
        // Reset graph first, then execute all statements fresh
        await get().resetGraph()
        const queries = splitQueries(get().editorContent)
        for (const q of queries) await get().executeQuery(q, true)
        await get().refreshGraph()
      },

      refreshGraph: async () => {
        try {
          const graph = await api.getGraph()
          set({ graph, expandedGroups: {} })
        } catch {
          toast.error('Server unreachable', { description: 'Could not fetch graph data' })
        }
      },

      resetGraph: async () => {
        try {
          await api.reset()
        } catch {
          toast.error('Reset failed', { description: 'Could not reset graph - is the server running?' })
        }
        set({
          graph: { nodes: [], edges: [] },
          results: [],
          highlightedNodeIds: new Set(),
          highlightedEdges: new Set(),
          activeResultId: null,
          expandedGroups: {},
        })
      },

      clearResults: () => set({ results: [], highlightedNodeIds: new Set(), highlightedEdges: new Set(), activeResultId: null }),

      selectResult: (id) => {
        const entry = get().results.find(r => r.id === id)
        if (!entry?.result) return
        // Toggle: clicking the active result clears highlights
        if (get().activeResultId === id) {
          set({ activeResultId: null, highlightedNodeIds: new Set(), highlightedEdges: new Set() })
          return
        }
        const { nodeIds, edgeKeys } = extractHighlights(entry.result)
        if (nodeIds.size > 0 || edgeKeys.size > 0) {
          set({ activeResultId: id, highlightedNodeIds: nodeIds, highlightedEdges: edgeKeys })
        } else {
          // Non-highlightable result: just mark active (blue border) without clearing existing highlights
          set({ activeResultId: id })
        }
      },

      clearHighlights: () => set({ activeResultId: null, highlightedNodeIds: new Set(), highlightedEdges: new Set() }),

      updateConfig: (partial) => {
        const prevLayoutMode = get().config.layoutMode
        set((s) => ({ config: { ...s.config, ...partial } }))
        // Reset collapse state on mode switch
        if (partial.layoutMode && partial.layoutMode !== prevLayoutMode) {
          set({ expandedGroups: {} })
        }
        // Sync server-side settings
        const serverUpdate: Record<string, number> = {}
        if (partial.ceilingMb !== undefined) serverUpdate.ceiling_mb = partial.ceilingMb
        if (partial.costThreshold !== undefined) serverUpdate.cost_threshold = partial.costThreshold
        if (Object.keys(serverUpdate).length > 0) {
          api.updateConfig(serverUpdate).catch(() => {
            toast.error('Server unreachable', { description: 'Could not update server config' })
          })
        }
      },

      toggleGroup: (parentId, kind) => set((s) => {
        const groups = { ...s.expandedGroups }
        const list = groups[parentId] || []
        if (list.includes(kind)) {
          groups[parentId] = list.filter(k => k !== kind)
        } else {
          groups[parentId] = [...list, kind]
        }
        return { expandedGroups: groups }
      }),

      resetExpandedGroups: () => set({ expandedGroups: {} }),
    }),
    {
      name: 'supergraph-storage',
      partialize: (state) => ({
        results: state.results,
        config: state.config,
        editorContent: state.editorContent,
      }),
      merge: (persisted, current) => {
        const p = persisted as Partial<SuperGraphState>
        const mergedConfig = { ...current.config, ...p?.config }
        // Migrate old 'force' value to 'cluster'
        if ((mergedConfig.layoutMode as string) === 'force') {
          mergedConfig.layoutMode = 'cluster'
        }
        // Clean up old viewMode from persisted state
        delete (mergedConfig as Record<string, unknown>).viewMode
        return {
          ...current,
          ...p,
          config: mergedConfig,
        }
      },
    }
  )
)
