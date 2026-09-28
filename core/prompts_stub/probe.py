"""Test stubs for probe prompts (no private content)."""

DECOMPOSE_STRANDS_SYSTEM = "STUB probe.DECOMPOSE_STRANDS_SYSTEM"


def generate_batch_system(batch_size: int) -> str:
    return f"STUB probe.generate_batch_system({batch_size})"


def evaluate_batch_system(batch_size: int) -> str:
    return f"STUB probe.evaluate_batch_system({batch_size})"
