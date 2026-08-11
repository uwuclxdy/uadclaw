#!/usr/bin/env bash
# deploy.sh — sync repo to a remote host and (re)build the docker compose stack.
# Manifest-diff pruning: only removes files that were tracked last deploy and have since
# been deleted from git. NEVER touches named volumes, data dirs, or logs.
#
# cp into: ./deploy.sh ; chmod +x ./deploy.sh
# Then set REMOTE (in .env) or pass as arg 1. REMOTE_DIR + COMPOSE_FILE override via .env.
#
# Dependencies (local): bash 4+, git, rsync, ssh, GNU coreutils ≥ 8.30 (for `sort -z`,
# `comm -z`). Remote: bash, xargs -0 (portable on GNU + BSD), rsync.

set -Eeuo pipefail

# --- allowlisted .env parse (no shell execution of .env contents) -------
# Only these keys are honoured; values failing the printable-no-meta regex are rejected.
# This is the intentional alternative to `set -a; . ./.env` — a hostile .env cannot inject
# commands, only set the allowlisted vars to safe strings.
if [ -f .env ]; then
  while IFS='=' read -r k v; do
    # Strip leading whitespace from the key so `  REMOTE=foo` parses (the old `case` arm
    # only matched `#` at column 0 — indented entries were silently dropped). Strip a
    # trailing CR from the value (CRLF-pasted .env).
    k="${k#"${k%%[![:space:]]*}"}"
    v="${v%$'\r'}"
    [[ "$v" =~ ^\"(.*)\"$ ]] && v="${BASH_REMATCH[1]}"
    [[ "$v" =~ ^\'(.*)\'$ ]] && v="${BASH_REMATCH[1]}"
    case "$k" in
      REMOTE|REMOTE_DIR|COMPOSE_FILE)
        # Tight allowlist — printable, no whitespace, no shell metas. Covers `user@host`,
        # `/srv/path-with.dots`, `docker-compose.prod.yml`, and rejects anything that could
        # break out of the remote shell once embedded.
        if ! [[ "$v" =~ ^[A-Za-z0-9._@/:-]+$ ]]; then
          echo "ERROR: .env value for $k='$v' contains characters outside [A-Za-z0-9._@/:-]" >&2
          exit 1
        fi
        export "$k=$v"
        ;;
      \#*|'') ;;  # comment / blank line
      *)
        # Log unknown keys so a typo (`RMOTE=…`) doesn't silently no-op.
        echo "WARN: ignoring unknown .env key '$k'" >&2
        ;;
    esac
  done < .env
fi

# --- config -------------------------------------------------------------
REMOTE="${1:-${REMOTE:-}}"
[ -n "$REMOTE" ] || { echo "usage: deploy.sh [user@host]  (or set REMOTE in .env)" >&2; exit 2; }

# Sanitise the default — basename of a path with spaces / unicode / `@` would fail the
# REMOTE_DIR regex below with a confusing error pointing at a path the user never typed.
REMOTE_DIR="${REMOTE_DIR:-/srv/$(basename "$PWD" | tr -c 'A-Za-z0-9._-' _)}"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"

# --- validate REMOTE / REMOTE_DIR shape ---------------------------------
# REMOTE must be `user@host`; REMOTE_DIR must be an absolute path with safe characters only.
# These regexes are the gate that lets us safely embed both into remote shell commands later.
[[ "$REMOTE"     =~ ^[A-Za-z0-9._-]+@[A-Za-z0-9.-]+$ ]] \
  || { echo "ERROR: REMOTE='$REMOTE' must match user@host" >&2; exit 1; }
[[ "$REMOTE_DIR" =~ ^/[A-Za-z0-9._/-]+$ ]] \
  || { echo "ERROR: REMOTE_DIR='$REMOTE_DIR' must be an absolute path with [A-Za-z0-9._/-] only" >&2; exit 1; }
[[ "$REMOTE_DIR" != *..* ]] || { echo "ERROR: REMOTE_DIR must not contain '..'" >&2; exit 1; }
[[ "$COMPOSE_FILE" =~ ^[A-Za-z0-9._/-]+$ ]] \
  || { echo "ERROR: COMPOSE_FILE='$COMPOSE_FILE' has unsafe characters" >&2; exit 1; }

# --- ssh / rsync transport options --------------------------------------
# BatchMode=yes refuses interactive prompts (no TOFU surprise in non-interactive contexts);
# ConnectTimeout caps hang on a misconfigured REMOTE; ServerAliveInterval keeps long rsyncs
# from being dropped by a NAT idle timer. Host-key onboarding is out of band — first deploy
# from a new machine needs `ssh-keyscan -t ed25519,rsa $host >> ~/.ssh/known_hosts` after
# verifying the fingerprint against an out-of-band source.
SSH_OPTS=(-o BatchMode=yes -o ConnectTimeout=10 -o ServerAliveInterval=30)
# rsync inherits ssh via $RSYNC_RSH; export so the rsync calls below pick up the same opts.
export RSYNC_RSH="ssh ${SSH_OPTS[*]}"

# --- sanity -------------------------------------------------------------
command -v rsync >/dev/null                                  || { echo "rsync missing locally"  >&2; exit 1; }
ssh "${SSH_OPTS[@]}" "$REMOTE" 'command -v docker >/dev/null' || { echo "docker missing on $REMOTE" >&2; exit 1; }

# --- manifest-diff prune (preserves volumes / data / logs) --------------
echo "→ computing manifest diff"
LOCAL_MANIFEST=$(mktemp)
REMOTE_MANIFEST=$(mktemp)
PRUNE_LIST=$(mktemp)
# Signal traps re-exit (with the conventional 128+signo code) so the EXIT trap then fires
# and removes the temp files; without the explicit `exit`, a Ctrl-C mid-command would clean
# up locals but let execution fall through to the next command (e.g. `docker compose up`).
trap 'rm -f "$LOCAL_MANIFEST" "$REMOTE_MANIFEST" "$PRUNE_LIST"' EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

# NUL-separated end-to-end so a filename with a newline doesn't split into two manifest lines.
git ls-files -z | sort -z > "$LOCAL_MANIFEST"

# Pass REMOTE_DIR via positional arg to a heredoc-streamed bash session — the variable is never
# interpolated into a quoted remote command, so a malicious-looking REMOTE_DIR (already regex-
# gated above) still cannot break out of the script.
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$REMOTE_DIR" <<'REMOTE_READ' | sort -z > "$REMOTE_MANIFEST"
set -Eeuo pipefail
cd "$1" 2>/dev/null || exit 0
[ -f .deploy-manifest ] && cat .deploy-manifest || true
REMOTE_READ

if [ -s "$REMOTE_MANIFEST" ]; then
  comm -z -23 "$REMOTE_MANIFEST" "$LOCAL_MANIFEST" > "$PRUNE_LIST"
  if [ -s "$PRUNE_LIST" ]; then
    echo "→ removing files dropped from git:"
    # NUL → newline for display only.
    tr '\0' '\n' < "$PRUNE_LIST" | sed 's/^/    /'
    # rsync the prune list to the remote, then ssh runs xargs over the staged file.
    # The previous `printf … | ssh … <<EOF` form was broken: the heredoc replaces ssh's
    # stdin so the pipe was silently discarded and xargs ran over an empty list.
    rsync -q "$PRUNE_LIST" "$REMOTE:$REMOTE_DIR/.deploy-prune-list"
    ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$REMOTE_DIR" <<'REMOTE_RM'
set -Eeuo pipefail
cd "$1"
# Cleanup trap fires on EXIT (success or failure) so an interrupted xargs doesn't
# leave a stale `.deploy-prune-list` for the next deploy to find.
trap 'rm -f .deploy-prune-list' EXIT
# xargs -0 works on both GNU and BSD; the upstream `[ -s "$PRUNE_LIST" ]` guard means
# the stream is never empty, so `-r` (GNU-only) isn't needed.
xargs -0 rm -f -- < .deploy-prune-list
REMOTE_RM
  fi
fi

# --- sync git-tracked files --------------------------------------------
echo "→ syncing git-tracked files to $REMOTE:$REMOTE_DIR"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$REMOTE_DIR" <<'REMOTE_MKDIR'
set -Eeuo pipefail
mkdir -p "$1"
REMOTE_MKDIR
git ls-files -z | rsync -av --from0 --files-from=- ./ "$REMOTE:$REMOTE_DIR/"

# --- persist manifest atomically ---------------------------------------
# Write to .deploy-manifest.new first, then mv into place — SIGINT mid-rsync leaves the
# previous manifest authoritative, so the next deploy doesn't read a truncated list and
# delete "missing" files. We rsync the sorted manifest because the pipe + heredoc form
# (`git ls-files | ssh ... cat > file`) was broken: the heredoc replaces ssh's stdin so
# the pipe is silently discarded and the file ends up empty.
echo "→ updating remote manifest atomically"
# If the local script dies between this rsync and the ssh-mv below, `.deploy-manifest.new`
# orphans on the remote; the next deploy's rsync overwrites it, so it self-heals.
rsync -q "$LOCAL_MANIFEST" "$REMOTE:$REMOTE_DIR/.deploy-manifest.new"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$REMOTE_DIR" <<'REMOTE_MANIFEST_MV'
set -Eeuo pipefail
cd "$1"
# Clean up the staged file if validation fails or anything below errors.
trap 'rm -f .deploy-manifest.new' EXIT
[ -s .deploy-manifest.new ] || { echo "ERROR: .deploy-manifest.new is empty — refusing to overwrite manifest" >&2; exit 1; }
mv .deploy-manifest.new .deploy-manifest
trap - EXIT
REMOTE_MANIFEST_MV

# --- (re)build + start --------------------------------------------------
echo "→ docker compose up -d --build"
ssh "${SSH_OPTS[@]}" "$REMOTE" bash -s -- "$REMOTE_DIR" "$COMPOSE_FILE" <<'REMOTE_COMPOSE'
set -Eeuo pipefail
cd "$1"
docker compose -f "$2" up -d --build
REMOTE_COMPOSE

echo "✓ deployed"
