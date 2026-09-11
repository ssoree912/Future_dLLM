"""Assert our OpenCompass prompt construction matches Sparse-dLLM's, exactly.

Tokenizer-only, so it runs on CPU in seconds. The reference path is lifted from
``Sparse_dLLM_DreamCausalLMInstruct.generate`` (their lines 225-242): flatten,
wrap as one user turn, apply the chat template, tokenize with special tokens.
A mismatch here silently costs accuracy on both the baseline and our row, and
would only show up as a bad number many GPU-hours later.

    python scripts/check_oc_prompt.py [--model PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from eval_oc.model import _convert_base_messages

MAX_SEQ_LEN = 2048


def reference_encode(tokenizer, inputs, max_seq_len):
    """Sparse-dLLM's path, batched exactly as they call it."""
    messages = _convert_base_messages(inputs)
    m = [[{"role": "user", "content": p}] for p in messages]
    formatted = tokenizer.apply_chat_template(m, add_generation_prompt=True,
                                              tokenize=False)
    return tokenizer(formatted, return_tensors="pt", padding=True,
                     truncation=True, add_special_tokens=True,
                     max_length=max_seq_len)["input_ids"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/Dream-v0-Instruct-7B")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    cases = {
        "piqa (plain string)":
            "Question: How do you open a jar?\nA. twist\nB. shout\nAnswer:",
        "mmlu 5-shot (long, hits truncation)":
            ("The following are multiple choice questions about professional "
             "law.\n\n" + "Question: filler clause of the statute.\nA. a\nB. b\n"
             "C. c\nD. d\nAnswer: A\n\n" * 400 +
             "Question: What is the holding?\nA. a\nB. b\nC. c\nD. d\nAnswer:"),
        "promptlist (role dicts)": [
            {"role": "HUMAN", "prompt": "Question: 2+2?\n"},
            {"role": "BOT", "prompt": "Answer: 4\n"},
            {"role": "HUMAN", "prompt": "Question: 3+3?\nAnswer:"},
        ],
        "empty-ish": "",
    }

    # One instance is enough: _encode and get_token_len are pure tokenizer work.
    model = object.__new__(
        __import__("eval_oc.model", fromlist=["DreamFutureOC"]).DreamFutureOC)
    model.tokenizer = tokenizer
    tokenizer.truncation_side = "left"

    failures = 0
    for name, case in cases.items():
        ref = reference_encode(tokenizer, [case], MAX_SEQ_LEN)
        ours = model._encode(_convert_base_messages([case])[0], MAX_SEQ_LEN)["input_ids"]
        ok = ref.shape == ours.shape and bool((ref == ours).all())
        print(f"  {'OK  ' if ok else 'FAIL'} {name:38s} "
              f"ref={tuple(ref.shape)} ours={tuple(ours.shape)}")
        if not ok:
            failures += 1
            print(f"       ref  {ref[0, :12].tolist()} ... {ref[0, -8:].tolist()}")
            print(f"       ours {ours[0, :12].tolist()} ... {ours[0, -8:].tolist()}")

    # Truncation side is a documented deviation: assert it actually bites, so
    # the long prompt keeps its question rather than its first few shots.
    long_case = cases["mmlu 5-shot (long, hits truncation)"]
    tokenizer.truncation_side = "right"
    right = reference_encode(tokenizer, [long_case], MAX_SEQ_LEN)
    tokenizer.truncation_side = "left"
    left = reference_encode(tokenizer, [long_case], MAX_SEQ_LEN)
    tail = "left-truncation keeps the final question"
    kept = tokenizer.decode(left[0, -40:])
    print(f"  {'OK  ' if 'holding' in kept else 'FAIL'} {tail:38s} "
          f"(right-truncated tail: {tokenizer.decode(right[0, -12:])!r})")
    failures += 0 if "holding" in kept else 1

    print("prompt construction matches Sparse-dLLM" if not failures
          else f"{failures} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
