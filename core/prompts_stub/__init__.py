"""Public stub prompts, used when the private submodule is not checked out.

Selected via ``LA_PROMPTS_STUB=1`` (see ``core.prompts``). Values are
intentionally obviously-fake so a test can never silently exercise real
prompt behavior.
"""

from . import clarify, language, material, plan, probe  # noqa: F401
