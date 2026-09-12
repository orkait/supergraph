import { useState } from 'react'
import { Dialog, DialogContent, DialogTitle } from '@/components/ui/dialog'
import { Button } from '@/components/ui/button'
import { ScrollArea } from '@/components/ui/scroll-area'
import { Separator } from '@/components/ui/separator'
import { categories, type Example } from '@/examples'
import { useSuperGraph } from '@/hooks/useSuperGraph'

interface ExamplesDialogProps {
  open: boolean
  onOpenChange: (open: boolean) => void
}

/**
 * Three-pane example picker built with pure flex + shadcn primitives.
 *
 * Why pure flex (no grid): nested grids inside a constrained-height
 * Dialog kept fighting each other for height, clipping the action
 * footer. Flex with explicit `min-h-0` / `min-w-0` on every parent
 * lets ScrollArea-backed children shrink under their content rather
 * than push siblings off the bottom.
 */
export function ExamplesDialog({ open, onOpenChange }: ExamplesDialogProps) {
  const setEditorContent = useSuperGraph((s) => s.setEditorContent)
  const resetGraph = useSuperGraph((s) => s.resetGraph)
  const refreshGraph = useSuperGraph((s) => s.refreshGraph)

  const [activeCategoryId, setActiveCategoryId] = useState(categories[0]?.id ?? '')
  const [activeExampleId, setActiveExampleId] = useState<string | null>(
    categories[0]?.examples[0]?.id ?? null,
  )

  const activeCategory = categories.find((c) => c.id === activeCategoryId) ?? categories[0]
  const activeExample =
    activeCategory.examples.find((e) => e.id === activeExampleId) ??
    activeCategory.examples[0]

  const loadExample = async (ex: Example) => {
    // Always wipe the graph view before loading. Carrying the previous
    // example's nodes around while the editor shows new code is a
    // confusing state. resetGraph() clears server-side state too,
    // ensuring the editor and the canvas agree.
    await resetGraph()
    setEditorContent(ex.script)
    // resetsGraph controls auto-replay of seed CREATE/UPSERT/ASSERT
    // statements (so the canvas pre-populates for demos with prebuilt
    // graphs). Examples whose value lies in the user manually running
    // each line keep this off and show only the editor content.
    if (ex.resetsGraph) {
      const queries = splitSeedStatements(ex.script)
      for (const q of queries) {
        await useSuperGraph.getState().executeQuery(q, true)
      }
    }
    await refreshGraph()
    onOpenChange(false)
  }

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent
        className="!max-w-[1200px] sm:!max-w-[1200px] w-[95vw] h-[80vh] p-0 overflow-hidden flex flex-col"
        showCloseButton={false}
      >
        <DialogTitle className="sr-only">Examples</DialogTitle>

        {/* Outer row: categories | (header + inner row + footer) */}
        <div className="flex flex-row flex-1 min-h-0 min-w-0">
          {/* Categories sidebar */}
          <aside className="w-[220px] shrink-0 border-r border-border bg-muted/30 flex flex-col min-h-0">
            <div className="px-3 pt-3 pb-1 text-[10px] font-semibold uppercase tracking-wide text-muted-foreground shrink-0">
              Categories
            </div>
            <ScrollArea className="flex-1 min-h-0">
              <div className="pb-2">
                {categories.map((c) => (
                  <button
                    key={c.id}
                    onClick={() => {
                      setActiveCategoryId(c.id)
                      setActiveExampleId(c.examples[0]?.id ?? null)
                    }}
                    className={
                      'w-full text-left px-3 py-1.5 text-sm transition-colors ' +
                      (c.id === activeCategoryId
                        ? 'bg-accent text-accent-foreground font-medium'
                        : 'hover:bg-accent/50')
                    }
                  >
                    {c.name}
                  </button>
                ))}
              </div>
            </ScrollArea>
          </aside>

          {/* Right column: header + inner row */}
          <section className="flex-1 flex flex-col min-w-0 min-h-0">
            {/* Header */}
            <div className="px-4 pt-3 pb-2 border-b border-border shrink-0">
              <div className="text-sm font-semibold">{activeCategory.name}</div>
              <div className="text-xs text-muted-foreground mt-0.5">
                {activeCategory.description}
              </div>
            </div>

            {/* Inner row: example list | detail (flex_1 with min) */}
            <div className="flex flex-row flex-1 min-h-0 min-w-0">
              {/* Example list */}
              <aside className="w-[280px] shrink-0 border-r border-border flex flex-col min-h-0">
                <ScrollArea className="flex-1 min-h-0">
                  <div>
                    {activeCategory.examples.map((ex) => (
                      <button
                        key={ex.id}
                        onClick={() => setActiveExampleId(ex.id)}
                        className={
                          'w-full text-left px-3 py-2 border-b border-border/50 transition-colors ' +
                          (ex.id === activeExample?.id
                            ? 'bg-accent text-accent-foreground'
                            : 'hover:bg-accent/40')
                        }
                      >
                        <div className="text-sm font-medium">{ex.name}</div>
                        <div className="text-[11px] text-muted-foreground mt-0.5 line-clamp-2">
                          {ex.description}
                        </div>
                      </button>
                    ))}
                  </div>
                </ScrollArea>
              </aside>

              {/* Detail panel: header (shrink-0) + scroll (flex-1) + footer (shrink-0) */}
              <article className="flex-1 flex flex-col min-w-0 min-h-0">
                {activeExample ? (
                  <>
                    {/* Detail header */}
                    <div className="px-4 pt-3 pb-2 shrink-0">
                      <div className="text-sm font-semibold">{activeExample.name}</div>
                      <div className="text-xs text-muted-foreground mt-1">
                        {activeExample.description}
                      </div>
                    </div>

                    {/* Script preview - scrolls in both axes */}
                    <div className="flex-1 min-h-0 min-w-0 mx-4 mb-3 border border-border rounded overflow-hidden">
                      <ScrollArea className="h-full">
                        <pre className="p-3 text-[11px] font-mono leading-relaxed whitespace-pre">
                          {activeExample.script}
                        </pre>
                      </ScrollArea>
                    </div>

                    {/* Footer (always visible because shrink-0 + parent has min-h-0) */}
                    <Separator />
                    <div className="px-4 py-3 flex flex-row items-center justify-end gap-2 shrink-0">
                      <Button variant="ghost" size="sm" onClick={() => onOpenChange(false)}>
                        Cancel
                      </Button>
                      <Button size="sm" onClick={() => loadExample(activeExample)}>
                        {activeExample.resetsGraph ? 'Load + reset graph' : 'Load into editor'}
                      </Button>
                    </div>
                  </>
                ) : (
                  <div className="flex-1 flex items-center justify-center text-muted-foreground text-sm">
                    Pick an example to preview.
                  </div>
                )}
              </article>
            </div>
          </section>
        </div>
      </DialogContent>
    </Dialog>
  )
}

/**
 * Lift seed-able write statements out of the example script for
 * auto-execution after reset. Reads (NODE / NODES / REMEMBER / etc.)
 * are NOT auto-run because they pollute the result panel before the
 * user has a chance to read the example. Multi-line BEGIN ... COMMIT
 * batches are kept intact.
 */
function splitSeedStatements(script: string): string[] {
  const lines = script.split('\n')
  const queries: string[] = []
  let batch: string[] = []
  let inBatch = false
  for (const line of lines) {
    const trimmed = line.trim()
    if (!trimmed || trimmed.startsWith('//')) continue
    if (trimmed === 'BEGIN') {
      inBatch = true
      batch = [line]
      continue
    }
    if (trimmed === 'COMMIT' && inBatch) {
      batch.push(line)
      queries.push(batch.join('\n'))
      batch = []
      inBatch = false
      continue
    }
    if (inBatch) {
      batch.push(line)
      continue
    }
    if (
      trimmed.startsWith('CREATE') ||
      trimmed.startsWith('UPSERT') ||
      trimmed.startsWith('ASSERT')
    ) {
      queries.push(trimmed)
    }
  }
  return queries
}
