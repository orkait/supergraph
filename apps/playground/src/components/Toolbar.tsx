import { Button } from '@/components/ui/button'
import { Separator } from '@/components/ui/separator'
import { Settings, BookOpen, Sun, Moon } from 'lucide-react'
import { useSuperGraph } from '@/hooks/useSuperGraph'
import { SettingsDialog } from '@/components/SettingsDialog'
import { ExamplesDialog } from '@/components/ExamplesDialog'
import { useState } from 'react'

export function Toolbar() {
  const isDark = useSuperGraph((s) => s.config.isDark)
  const updateConfig = useSuperGraph((s) => s.updateConfig)
  const [settingsOpen, setSettingsOpen] = useState(false)
  const [examplesOpen, setExamplesOpen] = useState(false)

  return (
    <>
      <div className="h-10 border-b border-border px-3 flex items-center gap-1.5 bg-card/50">
        <span className="text-sm font-semibold text-foreground mr-2">supergraph</span>
        <Separator orientation="vertical" className="h-5" />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 px-3 text-xs gap-1.5"
          onClick={() => setExamplesOpen(true)}
        >
          <BookOpen className="w-3.5 h-3.5" /> Examples
        </Button>
        <div className="flex-1" />
        <Button
          variant="ghost"
          size="sm"
          className="h-7 w-7 p-0"
          onClick={() => updateConfig({ isDark: !isDark })}
        >
          {isDark ? <Sun className="w-3.5 h-3.5" /> : <Moon className="w-3.5 h-3.5" />}
        </Button>
        <Button
          variant="ghost"
          size="sm"
          className="h-7 w-7 p-0"
          onClick={() => setSettingsOpen(true)}
        >
          <Settings className="w-3.5 h-3.5" />
        </Button>
      </div>
      <SettingsDialog open={settingsOpen} onOpenChange={setSettingsOpen} />
      <ExamplesDialog open={examplesOpen} onOpenChange={setExamplesOpen} />
    </>
  )
}
