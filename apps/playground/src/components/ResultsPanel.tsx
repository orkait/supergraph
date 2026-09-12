import { useSuperGraph, type ResultEntry } from '@/hooks/useSuperGraph'
import { Badge } from '@/components/ui/badge'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { ScrollArea } from '@/components/ui/scroll-area'
import { ChevronDown, ChevronRight } from 'lucide-react'
import { useState } from 'react'

function renderValue(value: unknown, depth = 0): React.ReactNode {
  if (value === null || value === undefined) return <span className="text-muted-foreground">null</span>
  if (typeof value !== 'object') return String(value)
  if (Array.isArray(value)) {
    if (value.length === 0) return <span className="text-muted-foreground">[]</span>
    if (typeof value[0] !== 'object') return value.join(', ')
    return <NestedKVTable data={Object.fromEntries(value.map((v, i) => [i, v]))} depth={depth} />
  }
  return <NestedKVTable data={value as Record<string, unknown>} depth={depth} />
}

function NestedKVTable({ data, depth = 0 }: { data: Record<string, unknown>; depth?: number }) {
  const keys = Object.keys(data)
  return (
    <table className={`w-full text-xs ${depth > 0 ? 'ml-1' : ''}`}>
      <tbody>
        {keys.map((k) => {
          const val = data[k]
          const isNested = val !== null && typeof val === 'object'
          return (
            <tr key={k} className="border-b border-border/50 align-top">
              <td className="p-1 text-muted-foreground w-32 whitespace-nowrap">{k}</td>
              <td className="p-1">{isNested ? renderValue(val, depth + 1) : String(val ?? '')}</td>
            </tr>
          )
        })}
      </tbody>
    </table>
  )
}

const HIGHLIGHTABLE_KINDS = new Set(['node', 'nodes', 'edges', 'path', 'paths', 'subgraph', 'match'])

function ResultCard({ entry }: { entry: ResultEntry }) {
  const [expanded, setExpanded] = useState(false)
  const activeResultId = useSuperGraph((s) => s.activeResultId)
  const selectResult = useSuperGraph((s) => s.selectResult)
  const showElapsed = useSuperGraph((s) => s.config.showElapsed)
  const isError = entry.error != null
  const isActive = activeResultId === entry.id
  const result = entry.result
  // canHighlight retained as a constant for future highlight-on-click
  // wiring; reference to satisfy TS no-unused-locals.
  void (result ? HIGHLIGHTABLE_KINDS.has(result.kind) : false)

  const renderTable = () => {
    if (!result?.data) return <span className="text-muted-foreground">No data</span>

    // Path: array of string IDs
    if (result.kind === 'path' && Array.isArray(result.data)) {
      return (
        <div className="text-xs flex items-center gap-1 flex-wrap">
          {result.data.map((id: string, i: number) => (
            <span key={i} className="flex items-center gap-1">
              <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary">{id}</span>
              {i < result.data.length - 1 && <span className="text-muted-foreground">→</span>}
            </span>
          ))}
        </div>
      )
    }

    // Paths: array of arrays of string IDs
    if (result.kind === 'paths' && Array.isArray(result.data)) {
      return (
        <div className="space-y-1">
          {result.data.map((path: string[], pi: number) => (
            <div key={pi} className="text-xs flex items-center gap-1 flex-wrap">
              <span className="text-muted-foreground mr-1">#{pi + 1}</span>
              {path.map((id: string, i: number) => (
                <span key={i} className="flex items-center gap-1">
                  <span className="px-1.5 py-0.5 rounded bg-primary/10 text-primary">{id}</span>
                  {i < path.length - 1 && <span className="text-muted-foreground">→</span>}
                </span>
              ))}
            </div>
          ))}
        </div>
      )
    }

    // Distance: single number
    if (result.kind === 'distance') {
      const d = result.data as number
      return <span className="text-sm">{d === -1 ? 'No path found' : `Distance: ${d}`}</span>
    }

    // Match: object with bindings and edges
    if (result.kind === 'match') {
      const bindings = Array.isArray(result.data?.bindings) ? result.data.bindings : []
      if (bindings.length === 0)
        return <span className="text-muted-foreground">No matches</span>
      const allKeys = new Set<string>()
      bindings.forEach((b: Record<string, string>) => Object.keys(b).forEach(k => allKeys.add(k)))
      const keys = [...allKeys]
      return (
        <table className="w-full text-xs">
          <thead>
            <tr className="border-b border-border">
              {keys.map(k => <th key={k} className="text-left p-1 text-muted-foreground">{k}</th>)}
            </tr>
          </thead>
          <tbody>
            {bindings.map((row: Record<string, string>, i: number) => (
              <tr key={i} className="border-b border-border/50">
                {keys.map(k => <td key={k} className="p-1">{row[k] || ''}</td>)}
              </tr>
            ))}
          </tbody>
        </table>
      )
    }

    // Subgraph: object with nodes and edges arrays
    if (result.kind === 'subgraph' && typeof result.data === 'object' && result.data !== null) {
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const nodes = Array.isArray((result.data as any).nodes) ? (result.data as any).nodes : []
      // eslint-disable-next-line @typescript-eslint/no-explicit-any
      const edges = Array.isArray((result.data as any).edges) ? (result.data as any).edges : []
      return (
        <div className="space-y-2">
          {nodes.length > 0 && (
            <div>
              <div className="text-xs font-semibold mb-1 text-muted-foreground">Nodes ({nodes.length})</div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    {nodes.length > 0 && Object.keys(nodes[0]).map((k) => (
                      <th key={k} className="text-left p-1 text-muted-foreground">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {nodes.map((row: Record<string, unknown>, i: number) => (
                    <tr key={i} className="border-b border-border/50">
                      {Object.keys(row).map((k) => (
                        <td key={k} className="p-1">{JSON.stringify(row[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
          {edges.length > 0 && (
            <div>
              <div className="text-xs font-semibold mb-1 text-muted-foreground">Edges ({edges.length})</div>
              <table className="w-full text-xs">
                <thead>
                  <tr className="border-b border-border">
                    {edges.length > 0 && Object.keys(edges[0]).map((k) => (
                      <th key={k} className="text-left p-1 text-muted-foreground">{k}</th>
                    ))}
                  </tr>
                </thead>
                <tbody>
                  {edges.map((row: Record<string, unknown>, i: number) => (
                    <tr key={i} className="border-b border-border/50">
                      {Object.keys(row).map((k) => (
                        <td key={k} className="p-1">{JSON.stringify(row[k])}</td>
                      ))}
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </div>
      )
    }

    // Array of objects (nodes, edges)
    if (Array.isArray(result.data)) {
      if (result.data.length === 0)
        return <span className="text-muted-foreground">Empty result</span>
      // Check first item is an object (not a string/number)
      if (typeof result.data[0] === 'object' && result.data[0] !== null) {
        const keys = Object.keys(result.data[0])
        return (
          <table className="w-full text-xs">
            <thead>
              <tr className="border-b border-border">
                {keys.map((k) => (
                  <th key={k} className="text-left p-1 text-muted-foreground">{k}</th>
                ))}
              </tr>
            </thead>
            <tbody>
              {result.data.map((row: Record<string, unknown>, i: number) => (
                <tr key={i} className="border-b border-border/50">
                  {keys.map((k) => (
                    <td key={k} className="p-1">{JSON.stringify(row[k])}</td>
                  ))}
                </tr>
              ))}
            </tbody>
          </table>
        )
      }
      // Array of primitives
      return <span className="text-sm">{JSON.stringify(result.data)}</span>
    }

    // Single object (stats, node, plan, etc.)
    if (typeof result.data === 'object') {
      return <NestedKVTable data={result.data as Record<string, unknown>} />
    }

    return <span className="text-sm">{JSON.stringify(result.data)}</span>
  }

  return (
    <div className={`border rounded-md mb-2 ${isActive ? 'border-primary bg-primary/5' : 'border-border bg-card/50'}`}>
      <div
        className="flex items-center gap-2 p-2 cursor-pointer hover:bg-accent/50"
        title={!isError && result ? 'Click to highlight · Ctrl+Click to expand' : 'Click to expand'}
        onClick={(e) => {
          if (e.ctrlKey || e.metaKey) {
            setExpanded(!expanded)
          } else if (!isError && result) {
            selectResult(entry.id)
          } else {
            setExpanded(!expanded)
          }
        }}
      >
        <span onClick={(e) => { e.stopPropagation(); setExpanded(!expanded) }} className="hover:text-foreground">
          {expanded ? <ChevronDown className="w-3 h-3" /> : <ChevronRight className="w-3 h-3" />}
        </span>
        <code className="text-xs text-muted-foreground truncate flex-1">
          {entry.query}
        </code>
        <Badge
          variant={isError ? 'destructive' : 'default'}
          className="text-[10px]"
        >
          {isError ? 'error' : result?.kind || 'ok'}
        </Badge>
        {result && (
          <span className="text-[10px] text-muted-foreground">
            {result.count}{showElapsed && <>&nbsp;&middot;&nbsp;{(result.elapsed_us / 1000).toFixed(1)}ms</>}
          </span>
        )}
      </div>
      {expanded && (
        <div className="p-2 border-t border-border">
          {isError ? (
            <div className="text-destructive text-xs font-mono">{entry.error}</div>
          ) : (
            <Tabs defaultValue="table" className="w-full">
              <TabsList className="h-7">
                <TabsTrigger value="table" className="text-xs h-6">Table</TabsTrigger>
                <TabsTrigger value="json" className="text-xs h-6">JSON</TabsTrigger>
              </TabsList>
              <TabsContent value="table" className="mt-2">{renderTable()}</TabsContent>
              <TabsContent value="json" className="mt-2">
                <pre className="text-xs text-foreground/80 whitespace-pre-wrap overflow-auto max-h-48">
                  {JSON.stringify(result, null, 2)}
                </pre>
              </TabsContent>
            </Tabs>
          )}
        </div>
      )}
    </div>
  )
}

export function ResultsPanel() {
  const results = useSuperGraph((s) => s.results)
  return (
    <div className="h-full flex flex-col bg-transparent">
      <div className="px-3 py-2 border-b flex-shrink-0 flex items-center bg-transparent">
        <div className="text-xs font-semibold text-muted-foreground">Results ({results.length})</div>
      </div>
      <div className="flex-1 overflow-hidden p-0">
        <ScrollArea className="h-full p-2">
          {results.map((entry) => (
            <ResultCard key={entry.id} entry={entry} />
          ))}
          {results.length === 0 && (
            <div className="text-muted-foreground/50 text-xs text-center mt-8">
              Run a query to see results
            </div>
          )}
        </ScrollArea>
      </div>
    </div>
  )
}
