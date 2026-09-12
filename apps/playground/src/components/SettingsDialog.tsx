import {
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
} from '@/components/ui/dialog'
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs'
import { Label } from '@/components/ui/label'
import { Switch } from '@/components/ui/switch'
import { Slider } from '@/components/ui/slider'
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select'
import { useSuperGraph, type LayoutDirection } from '@/hooks/useSuperGraph'

interface Props {
  open: boolean
  onOpenChange: (open: boolean) => void
}

export function SettingsDialog({ open, onOpenChange }: Props) {
  const config = useSuperGraph((s) => s.config)
  const updateConfig = useSuperGraph((s) => s.updateConfig)
  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogContent className="bg-card border-border text-foreground max-w-lg w-full p-6">
        <DialogHeader>
          <DialogTitle className="text-lg">Settings</DialogTitle>
        </DialogHeader>
        <Tabs defaultValue="graph" className="mt-2">
          <TabsList className="w-full">
            <TabsTrigger value="graph" className="flex-1">Graph</TabsTrigger>
            <TabsTrigger value="store" className="flex-1">Store</TabsTrigger>
            <TabsTrigger value="query" className="flex-1">Query</TabsTrigger>
          </TabsList>

          <TabsContent value="graph" className="space-y-5 mt-5">
            {config.layoutMode === 'dagre' && (
              <>
                <div className="space-y-2">
                  <Label className="text-sm">Layout Direction</Label>
                  <Select value={config.layoutDirection} onValueChange={(v) => updateConfig({ layoutDirection: v as LayoutDirection })}>
                    <SelectTrigger className="h-9"><SelectValue /></SelectTrigger>
                    <SelectContent>
                      <SelectItem value="TB">Top to Bottom</SelectItem>
                      <SelectItem value="LR">Left to Right</SelectItem>
                    </SelectContent>
                  </Select>
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm">Node Separation</Label>
                    <span className="text-sm font-mono text-muted-foreground">{config.nodesep}px</span>
                  </div>
                  <Slider
                    value={[config.nodesep]}
                    min={20} max={200} step={5}
                    onValueChange={(v) => updateConfig({ nodesep: Array.isArray(v) ? v[0] : v })}
                  />
                </div>
                <div className="space-y-3">
                  <div className="flex items-center justify-between">
                    <Label className="text-sm">Rank Separation</Label>
                    <span className="text-sm font-mono text-muted-foreground">{config.ranksep}px</span>
                  </div>
                  <Slider
                    value={[config.ranksep]}
                    min={30} max={300} step={10}
                    onValueChange={(v) => updateConfig({ ranksep: Array.isArray(v) ? v[0] : v })}
                  />
                </div>
              </>
            )}
            <div className="flex items-center justify-between py-1">
              <Label className="text-sm">Show Edge Labels</Label>
              <Switch
                checked={config.showEdgeLabels}
                onCheckedChange={(v) => updateConfig({ showEdgeLabels: v })}
              />
            </div>
            <div className="flex items-center justify-between py-1">
              <Label className="text-sm">Show Minimap</Label>
              <Switch checked={config.showMinimap} onCheckedChange={(v) => updateConfig({ showMinimap: v })} />
            </div>
          </TabsContent>

          <TabsContent value="store" className="space-y-5 mt-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Memory Ceiling</Label>
                <span className="text-sm font-mono text-muted-foreground">{config.ceilingMb} MB</span>
              </div>
              <Slider
                value={[config.ceilingMb]}
                min={64} max={1024} step={64}
                onValueChange={(v) => updateConfig({ ceilingMb: Array.isArray(v) ? v[0] : v })}
              />
              <p className="text-xs text-muted-foreground">Operations exceeding this limit will be rejected.</p>
            </div>
          </TabsContent>

          <TabsContent value="query" className="space-y-5 mt-5">
            <div className="space-y-3">
              <div className="flex items-center justify-between">
                <Label className="text-sm">Cost Threshold</Label>
                <span className="text-sm font-mono text-muted-foreground">{config.costThreshold.toLocaleString()}</span>
              </div>
              <Slider
                value={[config.costThreshold]}
                min={1000} max={1_000_000} step={1000}
                onValueChange={(v) => updateConfig({ costThreshold: Array.isArray(v) ? v[0] : v })}
              />
              <p className="text-xs text-muted-foreground">Maximum estimated frontier size before query rejection.</p>
            </div>
            <div className="flex items-center justify-between py-1">
              <Label className="text-sm">Show Elapsed Time</Label>
              <Switch checked={config.showElapsed} onCheckedChange={(v) => updateConfig({ showElapsed: v })} />
            </div>
          </TabsContent>
        </Tabs>
      </DialogContent>
    </Dialog>
  )
}
