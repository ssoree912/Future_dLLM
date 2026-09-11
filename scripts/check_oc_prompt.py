"""Assert our OpenCompass prompt construction matches Sparse-dLLM's, exactly.

Tokenizer-only, so it runs on CPU in seconds and needs no weights. Two
reference paths, because the two families are handled differently and mixing
them up is a mistake this repo has already made once:

  Dream  ``Sparse_dLLM_DreamCausalLMInstruct.generate`` (their lines 225-242):
         flatten, wrap as one user turn, apply the chat template, tokenize.
  LLaDA  ``Sparse_dLLM_LLaDACausalLM.generate`` (their lines 223-244): flatten
         and tokenize directly -- no chat template, even for the instruct
         checkpoint and even though their chat config passes no meta_template.

A mismatch here costs accuracy on the baseline and our row alike, and would
otherwise show up only as a bad number many GPU-hours later. For LLaDA it is
the only check available at all: no LLaDA weights or scorer exist in this repo,
so generation is never executed.

    python scripts/check_oc_prompt.py [--model PATH] [--llada PATH]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from transformers import AutoTokenizer

from eval_oc.llada_model import LLaDAFutureOC
from eval_oc.model import DreamFutureOC, _convert_base_messages

MAX_SEQ_LEN = 2048


def reference_encode(tokenizer, inputs, max_seq_len):
    """Sparse-dLLM's Dream path, batched exactly as they call it."""
    messages = _convert_base_messages(inputs)
    m = [[{"role": "user", "content": p}] for p in messages]
    formatted = tokenizer.apply_chat_template(m, add_generation_prompt=True,
                                              tokenize=False)
    return tokenizer(formatted, return_tensors="pt", padding=True,
                     truncation=True, add_special_tokens=True,
                     max_length=max_seq_len)["input_ids"]


def reference_encode_llada(tokenizer, inputs, max_seq_len):
    """Sparse-dLLM's LLaDA path: no chat template, straight to the tokenizer."""
    messages = _convert_base_messages(inputs)
    return tokenizer.batch_encode_plus(
        messages, return_tensors="pt", padding=True, truncation=True,
        add_special_tokens=True, max_length=max_seq_len)["input_ids"]


def _bare(cls, tokenizer):
    model = object.__new__(cls)
    model.tokenizer = tokenizer
    return model


def _compare(reference, model, tokenizer, cases):
    failures = 0
    for name, case in cases.items():
        ref = reference(tokenizer, [case], MAX_SEQ_LEN)
        ours = model._encode(_convert_base_messages([case])[0],
                             MAX_SEQ_LEN)["input_ids"]
        ok = ref.shape == ours.shape and bool((ref == ours).all())
        print(f"  {'OK  ' if ok else 'FAIL'} {name:38s} "
              f"ref={tuple(ref.shape)} ours={tuple(ours.shape)}")
        if not ok:
            failures += 1
            print(f"       ref  {ref[0, :12].tolist()} ... {ref[0, -8:].tolist()}")
            print(f"       ours {ours[0, :12].tolist()} ... {ours[0, -8:].tolist()}")
    return failures


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="model/Dream-v0-Instruct-7B")
    parser.add_argument("--llada", default="model/LLaDA-8B-Instruct-tokenizer")
    args = parser.parse_args()

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

    failures = 0
    print("Dream  (chat template, per dream_wrapper_instruct.py)")
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    tokenizer.truncation_side = "left"
    # Instantiating without __init__: _encode and get_token_len are pure
    # tokenizer work, so no weights need to be loaded to compare them.
    dream = _bare(DreamFutureOC, tokenizer)
    failures += _compare(reference_encode, dream, tokenizer, cases)

    print("LLaDA  (no chat template, per llada_wrapper.py)")
    llada_tok = AutoTokenizer.from_pretrained(args.llada, trust_remote_code=True)
    llada_tok.truncation_side = "left"
    llada = _bare(LLaDAFutureOC, llada_tok)
    failures += _compare(reference_encode_llada, llada, llada_tok, cases)

    # The two families must not converge: if LLaDA ever started emitting the
    # chat template, it would match Dream's shape and silently stop being the
    # baseline's prompt. Assert the difference is still there.
    plain = cases["piqa (plain string)"]
    templated = dream._encode(plain, MAX_SEQ_LEN)["input_ids"]
    raw = llada._encode(plain, MAX_SEQ_LEN)["input_ids"]
    untouched = llada_tok.decode(raw[0]) == plain
    print(f"  {'OK  ' if untouched else 'FAIL'} "
          f"{'LLaDA prompt stays untemplated':38s} "
          f"(dream {templated.shape[1]} tokens vs llada {raw.shape[1]})")
    failures += 0 if untouched else 1

    # Left truncation is OpenCompass's default (and so Sparse-dLLM's, inherited),
    # but this class sets it by hand because BaseModel loads no tokenizer. Assert
    # it actually bites: right truncation would drop the real question off a long
    # 5-shot prompt and keep the example shots instead.
    print("shared")
    long_case = cases["mmlu 5-shot (long, hits truncation)"]
    tokenizer.truncation_side = "right"
    right = reference_encode(tokenizer, [long_case], MAX_SEQ_LEN)
    tokenizer.truncation_side = "left"
    left = reference_encode(tokenizer, [long_case], MAX_SEQ_LEN)
    tail = "left truncation keeps the final question"
    kept = tokenizer.decode(left[0, -40:])
    print(f"  {'OK  ' if 'holding' in kept else 'FAIL'} {tail:38s} "
          f"(right-truncated tail: {tokenizer.decode(right[0, -12:])!r})")
    failures += 0 if "holding" in kept else 1

    print("prompt construction matches Sparse-dLLM" if not failures
          else f"{failures} mismatch(es)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
