import CodeMirror from '@uiw/react-codemirror'
import { supergraphLang, supergraphAutocomplete } from '@/lang/supergraph'
import { useSuperGraph } from '@/hooks/useSuperGraph'
import { keymap, type ViewUpdate, EditorView } from '@codemirror/view'
import { useCallback, useMemo } from 'react'
import { Button } from '@/components/ui/button'
import { Play, PlayCircle, Loader2, XCircle, Trash2, RotateCcw } from 'lucide-react'

function toggleComment(view: EditorView): boolean {
  const { state } = view
  const { from, to } = state.selection.main
  const fromLine = state.doc.lineAt(from).number
  const toLine = state.doc.lineAt(to).number

  const lines: { line: number; text: string }[] = []
  for (let i = fromLine; i <= toLine; i++) {
    const line = state.doc.line(i)
    lines.push({ line: i, text: line.text })
  }

  const allCommented = lines.every(l => l.text.trimStart().startsWith('//'))

  const changes = lines.map(l => {
    const line = state.doc.line(l.line)
    if (allCommented) {
      // Uncomment: remove first occurrence of "// " or "//"
      const idx = line.text.indexOf('//')
      const len = line.text[idx + 2] === ' ' ? 3 : 2
      return { from: line.from + idx, to: line.from + idx + len, insert: '' }
    } else {
      // Comment: prepend "// "
      return { from: line.from, to: line.from, insert: '// ' }
    }
  })

  view.dispatch({ changes })
  return true
}

export function EditorPanel() {
  const editorContent = useSuperGraph((s) => s.editorContent)
  const setEditorContent = useSuperGraph((s) => s.setEditorContent)
  const setEditorSelection = useSuperGraph((s) => s.setEditorSelection)
  const executeQuery = useSuperGraph((s) => s.executeQuery)
  const executeAll = useSuperGraph((s) => s.executeAll)
  const executeSelected = useSuperGraph((s) => s.executeSelected)
  const editorSelection = useSuperGraph((s) => s.editorSelection)
  const isDark = useSuperGraph((s) => s.config.isDark)
  const fontSize = useSuperGraph((s) => s.config.fontSize)
  const updateConfig = useSuperGraph((s) => s.updateConfig)
  const loading = useSuperGraph((s) => s.loading)
  const activeResultId = useSuperGraph((s) => s.activeResultId)
  const clearHighlights = useSuperGraph((s) => s.clearHighlights)
  const clearResults = useSuperGraph((s) => s.clearResults)
  const resetGraph = useSuperGraph((s) => s.resetGraph)
  const results = useSuperGraph((s) => s.results)
  const hasSelection = editorSelection.trim().length > 0

  const getFullLines = useCallback((view: EditorView) => {
    const { from, to } = view.state.selection.main
    if (from === to) return ''
    const firstLine = view.state.doc.lineAt(from)
    const lastLine = view.state.doc.lineAt(to)
    return view.state.sliceDoc(firstLine.from, lastLine.to)
  }, [])

  const handleRunSelected = useCallback(
    (view: EditorView) => {
      const text = getFullLines(view) || view.state.doc.lineAt(view.state.selection.main.head).text
      if (text.trim()) executeQuery(text.trim())
      return true
    },
    [executeQuery, getFullLines],
  )

  const onUpdate = useCallback(
    (update: ViewUpdate) => {
      if (!update.selectionSet) return
      const { from, to } = update.state.selection.main
      if (from === to) {
        setEditorSelection('')
        return
      }
      const firstLine = update.state.doc.lineAt(from)
      const lastLine = update.state.doc.lineAt(to)
      setEditorSelection(update.state.sliceDoc(firstLine.from, lastLine.to))
    },
    [setEditorSelection],
  )

  const fontTheme = useMemo(
    () => EditorView.theme({
      '&': { fontSize: `${fontSize}px` },
      '.cm-content': { fontFamily: 'inherit', padding: '2px 0' },
      '.cm-line': { padding: '0 4px' },
      '.cm-gutterElement': { padding: '0 6px 0 4px', lineHeight: 'inherit' },
    }),
    [fontSize],
  )

  const extensions = useMemo(
    () => [
      supergraphLang,
      supergraphAutocomplete,
      fontTheme,
      keymap.of([
        { key: 'Ctrl-Enter', run: handleRunSelected },
        { key: 'Ctrl-Shift-Enter', run: () => { executeAll(); return true } },
        { key: 'Ctrl-/', run: toggleComment },
      ]),
    ],
    [handleRunSelected, executeAll, fontTheme],
  )

  return (
    <div className="h-full flex flex-col bg-transparent">
      <div className="px-3 py-1.5 border-b bg-transparent flex-shrink-0 flex items-center justify-between">
        <div className="text-xs font-semibold text-muted-foreground">Editor</div>
        <div className="flex items-center gap-0.5 rounded-md border border-border bg-background shadow-sm overflow-hidden">
          <button
            className="h-6 w-6 flex items-center justify-center text-muted-foreground hover:bg-muted transition-colors select-none leading-none"
            onMouseDown={(e) => { e.preventDefault(); updateConfig({ fontSize: Math.max(10, fontSize - 1) }) }}
          >−</button>
          <span className="text-xs font-mono w-7 text-center text-foreground tabular-nums">{fontSize}</span>
          <button
            className="h-6 w-6 flex items-center justify-center text-muted-foreground hover:bg-muted transition-colors select-none leading-none"
            onMouseDown={(e) => { e.preventDefault(); updateConfig({ fontSize: Math.min(28, fontSize + 1) }) }}
          >+</button>
        </div>
      </div>
      <div className="flex-1 overflow-hidden p-0">
        <CodeMirror
          value={editorContent}
          onChange={setEditorContent}
          onUpdate={onUpdate}
          extensions={extensions}
          theme={isDark ? 'dark' : 'light'}
          height="100%"
          style={{ fontSize: `${fontSize}px` }}
          className="h-full [&_.cm-editor]:!h-full [&_.cm-editor]:!bg-transparent [&_.cm-scroller]:!overflow-auto [&_.cm-gutters]:!bg-transparent [&_.cm-gutters]:!text-muted-foreground [&_.cm-gutters]:!border-r [&_.cm-gutters]:!border-border [&_.cm-activeLineGutter]:!bg-muted/50 [&_.cm-activeLine]:!bg-muted/30 [&_.cm-selectionBackground]:!bg-primary/20"
          basicSetup={{
            lineNumbers: true,
            foldGutter: false,
            highlightActiveLine: true,
          }}
        />
      </div>
      <div className="px-3 py-1.5 flex items-center border-t bg-transparent flex-shrink-0 gap-1.5">
        {results.length > 0 && (
          <Button variant="ghost" size="sm" className="h-7 text-xs gap-1 text-[var(--btn-warn-text)] bg-[var(--btn-warn-bg)] hover:bg-[var(--btn-warn-hover)] border border-[var(--btn-warn-border)]" onMouseDown={(e) => { e.preventDefault(); clearResults() }}>
            <Trash2 className="w-3.5 h-3.5" /> Clear Results
          </Button>
        )}
        <Button variant="ghost" size="sm" className="h-7 text-xs gap-1 text-[var(--btn-danger-text)] bg-[var(--btn-danger-bg)] hover:bg-[var(--btn-danger-hover)] border border-[var(--btn-danger-border)]" onMouseDown={(e) => { e.preventDefault(); resetGraph() }}>
          <RotateCcw className="w-3.5 h-3.5" /> Reset DB
        </Button>
        <div className="flex-1" />
        {activeResultId && (
          <Button variant="ghost" size="sm" className="h-7 text-xs gap-1 text-[var(--btn-muted-text)] bg-[var(--btn-muted-bg)] hover:bg-[var(--btn-muted-hover)] border border-[var(--btn-muted-border)]" onMouseDown={(e) => { e.preventDefault(); clearHighlights() }}>
            <XCircle className="w-3.5 h-3.5" /> Clear Highlight
          </Button>
        )}
        <Button
          variant="ghost"
          size="sm"
          className="h-7 text-xs gap-1.5 text-[var(--btn-primary-text)] bg-[var(--btn-primary-bg)] hover:bg-[var(--btn-primary-hover)] border border-[var(--btn-primary-border)]"
          disabled={loading}
          onMouseDown={(e) => { e.preventDefault(); if (!loading) { if (hasSelection) executeSelected(); else executeAll() } }}
        >
          {loading
            ? <><Loader2 className="w-3.5 h-3.5 animate-spin" /> Running...</>
            : hasSelection
              ? <><Play className="w-3.5 h-3.5" /> Run Selected</>
              : <><PlayCircle className="w-3.5 h-3.5" /> Run All</>
          }
        </Button>
      </div>
    </div>
  )
}
