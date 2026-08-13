"""Model `CheckConstraint`s must equal the migrated database's, one for one.

`alembic check` (autogenerate) does not compare CHECK constraints, so the model and the
migrated database can drift with no gate: mutating a model `CheckConstraint` reds nothing
today. This test builds both sides from the live sources and asserts they are equal, so a
missing, extra, renamed or reworded constraint on either side reds.

Postgres re-renders a CHECK via `pg_get_constraintdef` as `CHECK (...)` with extra
parentheses and explicit type casts (`'reject'::text`), while the model stores the bare
literal. `normalise` canonicalises both sides to the token stream they share.
"""

import re

from sqlalchemy import CheckConstraint, text

from uadclaw import models  # noqa: F401  # registers every table on Base.metadata
from uadclaw.db import Base, get_engine

# A cast Postgres spells explicitly (`'reject'::text`) is invisible to this comparison for
# the same reason a parenthesis is: see `normalise`. Single-word types only — a multi-word
# type (`character varying`) is deliberately NOT matched, so it falls through and reds loudly
# rather than letting the `\s+` branch swallow a following keyword (`a::text OR b::text`).
_CAST_RE = re.compile(r"::[a-zA-Z_][a-zA-Z0-9_.]*")


def normalise(sql: str) -> str:
    """Canonicalise a CHECK body to the token stream both sides share.

    Postgres renders `CHECK (((a)::text <> 'x'::text))` where the model wrote `a <> 'x'`:
    a `CHECK (` prefix, redundant parentheses and explicit type casts. This strips all
    three, then uppercases identifiers/keywords and drops whitespace. Dropping every
    parenthesis is safe for this schema's constraints because none changes grouping
    without them (AND binds tighter than OR, and the comparisons are already
    parenthesised). Two bounds are stated rather than hidden: a drift that changes ONLY
    parenthesisation, or only a type cast, with an identical token sequence otherwise, is
    invisible here. A single-quoted literal is copied verbatim — case included, since a
    literal's case is part of its value (`'reject' <> 'REJECT'`) — so a literal that
    itself contains `::word` is not misread as a cast.
    """
    body = sql.strip()
    if body.upper().startswith("CHECK") and body[5:].lstrip().startswith("("):
        body = body[5:].lstrip()
    out = []
    i = 0
    n = len(body)
    while i < n:
        ch = body[i]
        if ch == "'":
            j = i + 1
            while j < n:
                if body[j] == "'":
                    if j + 1 < n and body[j + 1] == "'":
                        j += 2
                        continue
                    j += 1
                    break
                j += 1
            out.append(body[i:j])
            i = j
            continue
        if ch in "()":
            i += 1
            continue
        cast = _CAST_RE.match(body, i)
        if cast:
            i = cast.end()
            continue
        if not ch.isspace():
            out.append(ch.upper())
        i += 1
    return "".join(out)


def _model_constraints() -> set[tuple[str, str, str]]:
    """Every mapped table's `CheckConstraint`, as (table, name, normalised SQL).

    Enumerated from `Base.metadata` rather than from a hardcoded list so a constraint
    added in the future is auto-covered.
    """
    constraints = set()
    for table in Base.metadata.tables.values():
        for constraint in table.constraints:
            if isinstance(constraint, CheckConstraint):
                constraints.add((table.name, constraint.name, normalise(str(constraint.sqltext))))
    return constraints


async def _database_constraints() -> set[tuple[str, str, str]]:
    """Every CHECK constraint the migrated database actually holds on a mapped table.

    Introspected with the raw `pg_get_constraintdef(con.oid)` text rather than
    `inspect(engine).get_check_constraints`, whose dialect regex re-parses that same text
    and can mangle or warn on unusual forms; this hands `normalise` the authoritative
    rendering directly.
    """
    mapped = {table.name for table in Base.metadata.tables.values()}
    query = text(
        "SELECT c.relname, con.conname, pg_get_constraintdef(con.oid) "
        "FROM pg_constraint con "
        "JOIN pg_class c ON c.oid = con.conrelid "
        "JOIN pg_namespace n ON n.oid = c.relnamespace "
        "WHERE con.contype = 'c' AND n.nspname = 'public'"
    )
    async with get_engine().connect() as conn:
        rows = (await conn.execute(query)).all()
    return {
        (table, name, normalise(definition)) for table, name, definition in rows if table in mapped
    }


def _constraint_diff(model: set[tuple[str, str, str]], database: set[tuple[str, str, str]]) -> str:
    lines = []
    for label, side in (("model only", model - database), ("database only", database - model)):
        if side:
            lines.append(f"{label}:")
            lines.extend(
                f"  {table}.{name} ({normalised})" for table, name, normalised in sorted(side)
            )
    return "\n".join(lines)


async def test_model_check_constraints_match_migrated_database(db_env):
    # Liveness guard: without the `from uadclaw import models` side effect above,
    # `Base.metadata.tables` is empty and empty == empty passes vacuously.
    assert Base.metadata.tables, "Base.metadata has no tables: the models import was lost"
    model = _model_constraints()
    database = await _database_constraints()

    assert model == database, _constraint_diff(model, database)
