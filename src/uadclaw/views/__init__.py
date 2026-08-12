"""One module per dashboard screen, each owning its own router and its own templates.

Split this way so the screens do not share a file: `app.py` includes four routers and
nothing else, so a change to the corpus screen cannot touch the triage screen's routes.
"""
