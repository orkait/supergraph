# SuperGraph Playground (Under Development)

Interactive browser-based workbench for the supergraph DSL. Write queries, execute them, and visualize the graph in real-time.


## Quick Start

```bash
pip install supergraphdb[playground]
supergraph playground
```

Opens `http://localhost:7200` in your browser.

```bash
# Custom port
supergraph playground --port 8080

# Don't open browser
supergraph playground --no-browser
```

## Layout

Three resizable panels:

```
┌──────────────────────────────────────────────────────┐
│ [Examples ▾]  [Run All ▶]  [Reset]       [Settings]  │
├────────────────────┬─────────────────────────────────┤
│                    │                                  │
│   EDITOR           │   GRAPH (React Flow)             │
│   CodeMirror       │   Nodes colored by kind          │
│   DSL highlighting │   Labeled edges                  │
│   Ctrl+Enter: run  │   Drag to reposition             │
│                    │                                  │
├────────────────────┤   ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ ─ │
│                    │                                  │
│   RESULTS          │   Nodes: 12 │ Edges: 8 │ 0.3ms  │
│   Table / JSON     │                                  │
└────────────────────┴─────────────────────────────────┘
```

## Features

### Editor
- Custom syntax highlighting for the supergraph DSL
- **Ctrl+Enter** - run selected text (or current line)
- **Ctrl+Shift+Enter** - run all queries
- Supports `BEGIN...COMMIT` batch blocks
- Comments with `//`

### Graph Visualization
- Custom nodes showing `id` and `kind` badge
- Nodes colored by kind (function=blue, class=green, module=purple, etc.)
- Edges labeled with kind
- Auto-layout via dagre (hierarchical) with configurable direction (TB/LR)
- Tunable node separation and rank separation
- Drag nodes to reposition
- Pan, zoom, fit controls
- Optional minimap
- Wrapped in a card with inline node filter/search

### View Modes
- **Live** - graph updates after every mutation, always shows full graph
- **Query Result** - graph shows only nodes/edges returned by the last read query
- **Highlight** - full graph visible, query results highlighted with blue glow, rest dimmed

### Results Panel
- Stacked query results (newest on top)
- Expandable cards with **Table** and **JSON** views
- Nested tables for object values (e.g. `SYS STATS` output)
- Status badges (ok/error), elapsed time, result count
- Clear all results

### Persistence
- Editor content persists across page reloads (localStorage)
- Query results persist across reloads (capped at 50 entries)
- Theme (light/dark) and all layout settings persist across sessions
- Default example (Class Hierarchy) auto-loads and executes on first visit

### Settings
- **Graph** - view mode, layout algorithm, layout direction, node separation, rank separation, edge labels, minimap
- **Store** - memory ceiling (64–1024 MB)
- **Query** - cost threshold, explain-before-execute toggle

### Examples
Four pre-loaded example scripts accessible from the toolbar dropdown:

| Example | Nodes | Description |
|---|---|---|
| Function Call Graph | 5 | Diamond + chain call pattern |
| Class Hierarchy | 8 | extends, implements, uses edges |
| Code Graph | 16 | Functions, classes, modules with calls/extends/imports/contains |
| Microservices Map | 11 | Services, databases, queues with API calls and messaging |

Loading an example resets the graph, loads the script into the editor, and auto-executes all CREATE statements. The Class Hierarchy example loads by default on first visit.

## API

The playground exposes a REST API at `/api/`:

| Endpoint | Method | Description |
|---|---|---|
| `/api/execute` | POST | Execute a single DSL query |
| `/api/execute-batch` | POST | Execute multiple queries |
| `/api/graph` | GET | Get all nodes and edges |
| `/api/reset` | POST | Clear the graph |
| `/api/config` | POST | Update store configuration |

### Example

```bash
# Execute a query
curl -X POST http://localhost:7200/api/execute \
  -H 'Content-Type: application/json' \
  -d '{"query": "CREATE NODE \"a\" kind = \"x\" name = \"hello\""}'

# Get full graph
curl http://localhost:7200/api/graph
```

## Development

```bash
# Install backend deps
pip install -e ".[dev,playground]"

# Start backend
supergraph playground --no-browser --port 7200

# In another terminal, start frontend dev server
cd playground
npm install
npm run dev
```

The Vite dev server proxies `/api/*` to `http://localhost:7200`.

### Build

```bash
cd playground
npm run build
```

The built frontend at `playground/dist/` is served by the FastAPI server in production mode.

### Tech Stack

| Layer | Technology |
|---|---|
| Backend | FastAPI + uvicorn |
| Frontend | Vite + React 19 + TypeScript |
| Styling | Tailwind CSS v4 + shadcn/ui |
| Icons | Lucide React |
| Graph | React Flow v12 (@xyflow/react) |
| Editor | CodeMirror 6 (@uiw/react-codemirror) |
| Layout | dagre |
| State | Zustand |
