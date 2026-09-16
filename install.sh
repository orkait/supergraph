#!/bin/sh
# superclaw installer. Takes a machine from nothing to a working first run.
#   curl -LsSf https://raw.githubusercontent.com/orkait/supergraph/main/install.sh | sh
set -eu

REPO="${SUPERCLAW_REPO:-https://github.com/orkait/supergraph}"
REF="${SUPERCLAW_REF:-}"
PYTHON="${SUPERCLAW_PYTHON:-3.13}"
ASSUME_YES="${SUPERCLAW_YES:-}"
SOURCE=""
FORCE=""
WARM=1
STEPS=6

usage() {
    cat <<'EOF'
Install superclaw, with supergraph as its store.

Usage: install.sh [options]

  --ref <branch|tag|sha>   install this revision (default: the repo's default branch)
  --local <path>           install editable from a checkout instead of git
  --python <version>       interpreter for the tool environment (3.10 to 3.14, default 3.13)
  --force                  reinstall over an existing copy
  --no-warm                skip downloading the embedder; the first run downloads it instead
  -y, --yes                take the defaults, never prompt
  -h, --help               show this

Environment: SUPERCLAW_REPO, SUPERCLAW_REF, SUPERCLAW_PYTHON, SUPERCLAW_YES.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --ref) REF="${2:?--ref needs a value}"; shift 2 ;;
        --local) SOURCE="${2:?--local needs a path}"; shift 2 ;;
        --python) PYTHON="${2:?--python needs a value}"; shift 2 ;;
        --force) FORCE="--force"; shift ;;
        --no-warm) WARM=0; shift ;;
        -y|--yes) ASSUME_YES=1; shift ;;
        -h|--help) usage; exit 0 ;;
        *) printf 'install.sh: unknown option %s\n' "$1" >&2; usage >&2; exit 2 ;;
    esac
done

if [ -t 1 ] && [ -z "${NO_COLOR:-}" ] && [ "${TERM:-dumb}" != "dumb" ]; then
    BOLD=$(printf '\033[1m'); DIM=$(printf '\033[2m'); RED=$(printf '\033[31m')
    GREEN=$(printf '\033[32m'); YELLOW=$(printf '\033[33m'); OFF=$(printf '\033[0m')
else
    BOLD=""; DIM=""; RED=""; GREEN=""; YELLOW=""; OFF=""
fi

# curl | sh leaves stdin holding the script, so prompts read the terminal directly.
TTY=""
if [ -e /dev/tty ] && ( : >/dev/tty ) 2>/dev/null; then TTY=/dev/tty; fi

step=0
say() { printf '%s\n' "$*"; }
head_line() { step=$((step + 1)); printf '\n%s[%d/%d] %s%s\n' "$BOLD" "$step" "$STEPS" "$*" "$OFF"; }
ok() { printf '  %s+%s %s\n' "$GREEN" "$OFF" "$*"; }
warn() { printf '  %s!%s %s\n' "$YELLOW" "$OFF" "$*"; }
info() { printf '  %s%s%s\n' "$DIM" "$*" "$OFF"; }
die() { printf '\n%serror%s %s\n' "$RED" "$OFF" "$*" >&2; exit 1; }

confirm() {
    [ -n "$ASSUME_YES" ] && return 0
    [ -z "$TTY" ] && return 0
    printf '\n%s [Y/n] ' "$1"
    read -r reply < "$TTY" || return 0
    case "$reply" in [nN]*) return 1 ;; *) return 0 ;; esac
}

printf '%s\n' "$BOLD"
cat <<'EOF'
   superclaw
EOF
printf '%s' "$OFF"
say "${DIM}a terminal coding agent that remembers, with supergraph as its store${OFF}"

head_line "Checking this machine"
OS=$(uname -s 2>/dev/null || echo unknown)
case "$OS" in
    Linux|Darwin) ok "$OS $(uname -m)" ;;
    *) die "$OS is not supported; superclaw needs Linux or macOS" ;;
esac
command -v curl >/dev/null 2>&1 || command -v wget >/dev/null 2>&1 || die "curl or wget is required"

if command -v uv >/dev/null 2>&1; then
    ok "uv $(uv --version 2>/dev/null | cut -d' ' -f2)"
    NEED_UV=0
else
    warn "uv is missing; the installer will fetch it"
    NEED_UV=1
fi

if command -v superclaw >/dev/null 2>&1; then
    warn "superclaw is already installed at $(command -v superclaw); it will be replaced"
    FORCE="--force"
fi

if command -v bwrap >/dev/null 2>&1; then
    ok "bubblewrap present, so bash runs sandboxed"
else
    warn "bubblewrap missing, so bash will run unsandboxed"
    [ "$OS" = "Linux" ] && info "apt install bubblewrap, or dnf install bubblewrap"
fi

MISSING=""
for tool in rg fd jq gh; do
    command -v "$tool" >/dev/null 2>&1 || MISSING="$MISSING $tool"
done
[ -n "$MISSING" ] && info "optional tools not found:$MISSING (superclaw falls back to slower built-ins)"

head_line "Planning the install"
if [ -n "$SOURCE" ]; then
    [ -f "$SOURCE/pyproject.toml" ] || die "no pyproject.toml in $SOURCE; point --local at the repository root"
    TARGET="$SOURCE (editable)"
else
    TARGET="$REPO"
    [ -n "$REF" ] && TARGET="$REPO @ $REF"
fi
info "package  supergraphdb[superclaw], which brings supergraph"
info "source   $TARGET"
info "python   $PYTHON"
info "tools    superclaw, supergraph, supergraph-mcp"
confirm "Install now?" || { say "Nothing was installed."; exit 0; }

head_line "Installing"
if [ "$NEED_UV" = "1" ]; then
    curl -LsSf https://astral.sh/uv/install.sh | sh >/dev/null 2>&1 || die "could not install uv"
    # shellcheck disable=SC1090
    [ -f "$HOME/.local/bin/env" ] && . "$HOME/.local/bin/env"
    PATH="$HOME/.local/bin:$PATH"; export PATH
    command -v uv >/dev/null 2>&1 || die "uv installed but not on PATH; add \$HOME/.local/bin to PATH and retry"
    ok "uv installed"
fi

if [ -n "$SOURCE" ]; then
    # shellcheck disable=SC2086
    uv tool install --python "$PYTHON" $FORCE -e "$SOURCE[superclaw]" || die "install failed"
else
    GIT="$REPO"
    [ -n "$REF" ] && GIT="$REPO@$REF"
    # shellcheck disable=SC2086
    uv tool install --python "$PYTHON" $FORCE "supergraphdb[superclaw] @ git+$GIT" || die "install failed"
fi
command -v superclaw >/dev/null 2>&1 || die "installed, but superclaw is not on PATH; run: uv tool update-shell"
ok "superclaw installed at $(command -v superclaw)"

head_line "Verifying"
superclaw --help >/dev/null 2>&1 || die "superclaw will not start; run superclaw --help to see why"
ok "starts"
DOCTOR=$(superclaw doctor 2>/dev/null || true)
KEYED=$(printf '%s' "$DOCTOR" | sed -n 's/^providers with a key: //p')
MODEL=$(printf '%s' "$DOCTOR" | sed -n 's/^model \([^ ]*\).*/\1/p')
[ -n "$MODEL" ] && ok "model $MODEL"
printf '%s' "$DOCTOR" | grep -q '^sandbox on' && ok "sandbox on" || warn "sandbox off"

if [ "$WARM" = "1" ]; then
    head_line "Warming the store"
    info "downloading the default embedder (model2vec, about 30 MB) so the first run is not slow"
    if superclaw repo-map >/dev/null 2>&1 && superclaw skills >/dev/null 2>&1; then
        ok "store ready"
    else
        warn "could not warm the store; the first run will do it"
    fi
else
    head_line "Warming the store"
    info "skipped; the first run downloads the embedder"
fi

head_line "Provider key"
if [ -n "$KEYED" ]; then
    ok "keys found for: $KEYED"
else
    warn "no provider key yet, so superclaw cannot reach a model"
    if [ -n "$TTY" ] && [ -z "$ASSUME_YES" ] && confirm "Run superclaw setup now?"; then
        superclaw setup < "$TTY" || warn "setup did not finish; run superclaw setup when ready"
    else
        info "run: superclaw setup"
        info "or export a key, for example OPENROUTER_API_KEY"
    fi
fi

printf '\n%sReady.%s\n\n' "$BOLD" "$OFF"
say "  cd your-project"
say "  superclaw                       start the terminal UI"
say "  superclaw --mode auto exec \"…\"  one-shot, no UI"
say "  superclaw doctor                model, store, sandbox, tools"
say ""
say "${DIM}Full-screen by default. /tui default keeps your scrollback instead.${OFF}"
