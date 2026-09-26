"""Synthetic passkey prompt used by the speed scripts' quality gate.

The haystack is made of unique numbered notes, so the needle is the only
occurrence of the pass key; the prompt is built directly in token ids so that
its length is exactly ``ctx`` tokens.
"""

from __future__ import annotations

DEFAULT_NEEDLE = "42891735"


def make_passkey_prompt(tokenizer, ctx: int, needle: str = DEFAULT_NEEDLE) -> tuple[str, list[int]]:
    prefix = (
        "You are a careful reader. A unique pass key is hidden once in the notes. "
        "When asked, repeat only the pass key digits.\n"
    )
    needle_text = f"\n*** The pass key is {needle}. Remember it. ***\n"
    suffix = "\nQuestion: What is the pass key? Answer with digits only.\nThe pass key is"
    prefix_ids = tokenizer.encode(prefix, add_special_tokens=True)
    needle_ids = tokenizer.encode(needle_text, add_special_tokens=False)
    suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
    room = ctx - len(prefix_ids) - len(needle_ids) - len(suffix_ids)
    if room < 8:
        raise ValueError(f"ctx={ctx} too small for passkey template")
    pre_h = room // 2
    post_h = room - pre_h

    def _unique_hay(n_tokens: int, start: int) -> list[int]:
        out: list[int] = []
        i = start
        while len(out) < n_tokens:
            piece = tokenizer.encode(
                f"Note {i}: river-{i} stays calm and the archive id is {i}. ",
                add_special_tokens=False,
            )
            out.extend(piece)
            i += 1
        return out[:n_tokens]

    ids = prefix_ids + _unique_hay(pre_h, 0) + needle_ids + _unique_hay(post_h, 10_000) + suffix_ids
    ids = ids[:ctx]
    text = tokenizer.decode(ids, skip_special_tokens=False)
    return text, ids


__all__ = ["DEFAULT_NEEDLE", "make_passkey_prompt"]
