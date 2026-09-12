import { StreamLanguage } from '@codemirror/language'
import { autocompletion, CompletionContext } from '@codemirror/autocomplete'

const KEYWORDS = new Set([
  'CREATE', 'NODE', 'NODES', 'EDGE', 'EDGES', 'WHERE', 'FROM', 'TO',
  'TRAVERSE', 'MATCH', 'DELETE', 'UPDATE', 'UPSERT', 'INCREMENT',
  'PATH', 'PATHS', 'SHORTEST', 'DISTANCE', 'ANCESTORS', 'DESCENDANTS',
  'COMMON', 'NEIGHBORS', 'SUBGRAPH', 'DEPTH', 'MAX_DEPTH', 'LIMIT',
  'SET', 'BY', 'OF', 'AND', 'OR', 'NOT', 'BEGIN', 'COMMIT',
  'KIND', 'REGISTER', 'UNREGISTER', 'DESCRIBE', 'STATS', 'EXPLAIN',
  'CHECKPOINT', 'REBUILD', 'INDICES', 'CLEAR', 'WAL', 'REQUIRED',
  'OPTIONAL', 'SINCE', 'SLOW', 'FREQUENT', 'FAILED', 'KINDS',
  'STATUS', 'REPLAY', 'LOG', 'CACHE', 'NULL', 'INDEGREE', 'OUTDEGREE',
])

export const supergraphLang = StreamLanguage.define({
  token(stream) {
    if (stream.eatSpace()) return null

    // Comments
    if (stream.match('//')) {
      stream.skipToEnd()
      return 'comment'
    }

    // Strings
    if (stream.match('"')) {
      while (!stream.eol()) {
        const ch = stream.next()
        if (ch === '\\') { stream.next(); continue }
        if (ch === '"') break
      }
      return 'string'
    }

    // Numbers
    if (stream.match(/-?[0-9]+(\.[0-9]+)?/)) return 'number'

    // Arrow operators
    if (stream.match('->')) return 'operator'
    if (stream.match('-[')) return 'operator'
    if (stream.match(']->')) return 'operator'

    // Multi-char comparison operators (must come before single-char)
    if (stream.match('!=') || stream.match('>=') || stream.match('<=')) return 'operator'
    if (stream.match(/[=><]/)) return 'operator'

    // Words (keywords and identifiers)
    if (stream.match(/[a-zA-Z_][a-zA-Z0-9_]*/)) {
      const word = stream.current()
      if (word === 'SYS') return 'keyword2'
      if (KEYWORDS.has(word)) return 'keyword'
      return 'variableName'
    }

    stream.next()
    return null
  },
})

const dslOptions = Array.from(KEYWORDS).map(k => ({ label: k, type: 'keyword' }))

function dslCompletion(context: CompletionContext) {
  const word = context.matchBefore(/\w*/)
  if (word?.from === word?.to && !context.explicit) return null
  return {
    from: word ? word.from : context.pos,
    options: dslOptions
  }
}

export const supergraphAutocomplete = autocompletion({ override: [dslCompletion] })
