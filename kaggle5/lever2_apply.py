# RETIRED — this is not the applier any more.
#
# The lever patch was re-based onto the current TAAF bundle (dataset v1,
# 2026-09-28) and now carries lever-2 (.grid) *and* lever-3 (deterministic
# episodic memory + hard prune gate). Apply it with:
#
#     python lever3_apply.py
#
# lever3_apply.py hash-verifies all five post-images and proves the capability
# markers, then honours VARIANT=lever3|kernel. This file stays only so a stale
# caller fails loudly instead of applying an obsolete 4-hunk patch.
raise SystemExit(
    "lever2_apply.py is retired. Use lever3_apply.py (applies + verifies the "
    "re-based lever patch: .grid + episodic memory + prune gate)."
)
