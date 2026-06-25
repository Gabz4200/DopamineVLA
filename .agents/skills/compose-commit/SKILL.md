---
name: compose-commit
description: 'Compose multiple clean commits from a batch of uncommitted changes. Groups files by logical scope using diff analysis, then stages and commits each group separately with conventional commit messages. Use when user has many changed files across different areas and wants organized commits, mentions "compose commits", "split into commits", "multi-commit", "commit composer", or has mixed changes (features + fixes + refactors in one batch). Runs `git status` to discover changes, groups them by type/scope, and commits each group individually. Unlike git-commit (single commit), this creates N separate commits from N logical groups.'
license: MIT
allowed-tools: Bash
---

# Commit Composer

Create multiple clean, focused commits from a single batch of uncommitted changes. Automatically groups related changes by logical scope and commits each group separately with appropriate conventional commit messages.

## When to use

- User has many changed files across different subsystems
- Changes mix multiple concerns (feature + fix + refactor + docs)
- User says "compose commits", "split this into commits", "multi-commit"
- User has unrelated changes in the same working tree and wants organized history

## How it works

1. **Discover** — `git status` all changed/untracked files
2. **Analyze** — read each file's diff to determine type (feat/fix/refactor/etc.) and scope
3. **Group** — cluster files by logical concern (same type + same area)
4. **Present** — show proposed groups for user approval/adjustment
5. **Commit** — stage and commit each group as a separate conventional commit

## Workflow

### 1. Discover all changes

```bash
git status --porcelain
```

Note staged vs unstaged vs untracked files.

### 2. Analyze each changed file

For each file, read a sample of its diff to classify:

```bash
git diff -- path/to/file
# or for staged files:
git diff --staged -- path/to/file
```

**Classification heuristics per file:**

| Signal | Type |
|--------|------|
| New functions, exports, components, features, API endpoints | `feat` |
| Bug-related comments, edge-case handling, guard clauses | `fix` |
| Moving code, renaming, extracting functions, no behavior change | `refactor` |
| Docstrings, comments, `*.md`, README | `docs` |
| Formatting only, whitespace, import sorting | `style` |
| `test_*`, `*_test.*`, `*.test.*`, `*_spec.*`, `*Spec.*` | `test` |
| Config files, deps, CI, workflows, `pyproject.toml`, `package.json` | `chore` |
| Performance-related changes (caching, algorithms, data structures) | `perf` |

**Scope inference:**
- Directory/module name from the file path (e.g. `src/auth/` → scope `auth`)
- Root-level config/doc files scope to the project name or no scope

### 3. Group files by logical concern

Group files that share the **same type** AND **same or adjacent scope**. Default groups:

```
Group 1: feat(auth) — login.ts, register.ts, middleware.ts
Group 2: fix(api) — users.ts, posts.ts, validation.ts
Group 3: test(auth) — test_login.py, test_middleware.py
Group 4: chore(deps) — pyproject.toml, Cargo.toml
```

**Rules:**
- NEVER split a single file across two commits
- DO split same directory if changes have different types (e.g. `feat` + `fix` in same dir → two groups)
- Files with no clear logic change (formatting, trivial) merge into the nearest group or `chore`
- Untracked new files that look like features → `feat` group
- Test files → pair with their feature group, or separate `test(scope)` group if numerous

### 4. Present groups to user

Show the proposed commit plan:

```bash
# Example output:
# ┌──────────────────────────────────────────────────┐
# │ Proposed commits:                                │
# │                                                  │
# │ 1) feat(auth): add OAuth2 login flow             │
# │    src/auth/login.ts                             │
# │    src/auth/register.ts                          │
# │                                                  │
# │ 2) fix(api): handle empty user list edge case    │
# │    src/api/users.ts                              │
# │                                                  │
# │ 3) test(auth): cover OAuth2 flow                 │
# │    tests/test_auth.py                            │
# │                                                  │
# │ 4) chore(deps): update pyproject.toml             │
# │    pyproject.toml                                 │
# │                                                  │
# │ a) accept / e) edit / c) cancel ?                │
# └──────────────────────────────────────────────────┘
```

Ask, don't assume. Use `ask` tool with options.

Options to offer:
- **Accept all** — proceed with proposed groups
- **Edit groups** — user specifies regrouping (e.g. "move users.ts to group 1", "merge groups 2 and 3")
- **Cancel** — do nothing

### 5. Stage and commit each group

Process groups in order, one commit at a time:

```bash
# Stage group files
git add file1 file2 file3

# Commit with conventional message
git commit -m "$(cat <<'EOF'
<type>(<scope>): <description>

<optional body — summarize what was done>
EOF
)"
```

**Per-commit rules:**
- Description under 72 chars, imperative mood, present tense
- Body (if needed) explains WHY, not what (the diff shows what)
- Only the files for this group are staged — no stragglers
- Verify staging before each commit: `git diff --staged --name-only`

## Examples

### Mixed feature + fix + refactor

```
M  src/auth/login.py       # +oauth2_support() → feat(auth)
M  src/auth/middleware.py   # +rate_limiting()  → feat(auth)
M  src/api/users.py         # fix null pointer   → fix(api)
M  src/api/posts.py         # refactor queries   → refactor(api)
M  tests/test_auth.py       # new tests          → test(auth)
```

**Proposed:**
1. `feat(auth): add OAuth2 login flow and rate limiting`
2. `fix(api): guard against null user in list endpoint`
3. `refactor(api): extract post query builder`
4. `test(auth): add OAuth2 and rate limit coverage`

### Config + lockfile update alongside real work

```
M  pyproject.toml            # dep bump    → chore(deps)
M  uv.lock                   # lock update  → chore(deps)
M  src/core/engine.py        # perf tweak   → perf(core)
M  src/core/cache.py         # perf tweak   → perf(core)
```

**Proposed:**
1. `perf(core): optimize cache eviction in engine`
2. `chore(deps): update project dependencies`

## Best practices

- **One logical change per commit** — the whole point of this skill
- **Test files with their feature** unless there are many, then separate `test(scope)`
- **Order commits logically** — foundation changes first (refactor, chore), then features/fixes
- **Never commit secrets** — `.env`, credentials, keys, tokens
- **Commit message quality** — each message must make sense independently in `git log`
- **Verify each commit** — `git diff --staged --name-only` before committing to avoid mis-staging

## Git Safety

- NEVER update git config
- NEVER run destructive commands (`--force`, hard reset) without explicit request
- NEVER skip hooks (`--no-verify`)
- If a commit fails (hooks, conflicts), stop and report; don't retry blindly
- ALWAYS verify diff before each commit
