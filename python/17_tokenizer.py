"""
Example 17: Tokenizer
Use the engine's tokenize and detokenize methods to count tokens, inspect
how text is encoded, and build a simple token budget before sending a prompt.
"""

import litert_lm

MODEL_PATH = "gemma-4-E4B-it.litertlm"

litert_lm.set_min_log_severity(litert_lm.LogSeverity.ERROR)

TOKEN_BUDGET = 50

samples = [
    "Hello, world!",
    "The quick brown fox jumps over the lazy dog.",
    "LiteRT-LM runs large language models completely on-device.",
    "Tokenisation splits text into the atomic units a model actually sees.",
]

with litert_lm.Engine(MODEL_PATH) as engine:
    #  Part A: inspect token ids
    print("=== Part A: encoding ===")
    for text in samples:
        ids = engine.tokenize(text)
        recovered = engine.detokenize(ids)
        print(f"Text    : {text!r}")
        print(f"Token ids ({len(ids)}): {ids}")
        print(f"Decoded : {recovered!r}")
        print()

    #  Part B: token budget guard
    print(f"=== Part B: budget guard (limit = {TOKEN_BUDGET} tokens) ===")
    long_prompt = " ".join(["word"] * 80)
    ids = engine.tokenize(long_prompt)
    if len(ids) > TOKEN_BUDGET:
        truncated_ids = ids[:TOKEN_BUDGET]
        truncated_text = engine.detokenize(truncated_ids)
        print(f"Prompt was {len(ids)} tokens — truncated to {len(truncated_ids)}.")
        print(f"Truncated text: {truncated_text!r}")
    else:
        print(f"Prompt fits within budget ({len(ids)} tokens).")
