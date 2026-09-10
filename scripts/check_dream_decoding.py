"""Short GPU comparison against Sparse-dLLM's actual diffusion_generate API.

Checks two schedules, every post-seed state, final tokens, and RNG consumption.
The optional scorer check uses random weights and checks execution, not quality.
No teacher labels are written by this command.
"""

import argparse
import gc
import importlib
import json
from pathlib import Path
import sys
import types

import torch
from transformers import AutoTokenizer

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(REPO / "model/Dream-v0-Instruct-7B"))
    parser.add_argument("--reference-root", default=str(REPO.parent / "Sparse-dLLM"))
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    reference_path = Path(args.reference_root) / "opencompass/models/sparse_dllm/dream"
    package = types.ModuleType("_sparse_dream_reference")
    package.__path__ = [str(reference_path)]
    sys.modules[package.__name__] = package
    ref_config = importlib.import_module(package.__name__ + ".configuration_dream")
    ref_model = importlib.import_module(package.__name__ + ".modeling_dream")

    config = ref_config.DreamConfig.from_pretrained(args.model)
    config.block_len, config.keep_ratio, config.kernel_size = 32, 1.0, 3
    config.use_cache = False
    model = ref_model.DreamModel.from_pretrained(
        args.model, config=config, torch_dtype=torch.bfloat16, device_map="cuda:0").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "A box has 12 red balls and 8 blue balls. "
          "How many balls are there in total? Explain briefly."}],
        add_generation_prompt=True, return_tensors="pt").to("cuda:0")
    reference = []
    for steps in (64, 32):
        torch.manual_seed(0)
        output = model.diffusion_generate(
            prompt, max_new_tokens=64, steps=steps, alg="entropy",
            temperature=0.2, top_p=0.95, return_dict_in_generate=True, output_history=True)
        reference.append((output.sequences.cpu(), [x.cpu() for x in output.history],
                          torch.cuda.get_rng_state().cpu()))
        print(f"reference steps={steps} complete", flush=True)
    del output, model
    gc.collect()
    torch.cuda.empty_cache()

    from future_dllm import load_model, PromptUtilityStudent, StudentConfig
    from future_dllm.dream_decoding import DreamDecoding, require_matching_decoding
    model, backend = load_model(args.model, max_seq_len=2048, block_length=32)
    report = {"decoding": DreamDecoding().metadata(), "seed": 0,
              "reference_root": str(reference_path), "cases": []}
    for steps, (expected, history, rng) in zip((64, 32), reference):
        actual_history = []

        def observe(x, block, step):
            if step:
                actual_history.append(x.cpu().clone())

        torch.manual_seed(0)
        actual = backend.generate(model, prompt, gen_length=64, steps=steps,
                                  on_step=observe)
        case = {
            "gen_length": 64, "steps": steps, "keep_ratio": 1.0,
            "tokens_equal": torch.equal(expected, actual.cpu()),
            "history_equal": len(history) == len(actual_history) and all(
                torch.equal(a, b) for a, b in zip(history, actual_history)),
            "rng_equal": torch.equal(rng, torch.cuda.get_rng_state().cpu()),
            "remaining_masks": int((actual[:, prompt.shape[1]:] == backend.mask_id).sum()),
            "text": tokenizer.decode(actual[0, prompt.shape[1]:], skip_special_tokens=True),
        }
        report["cases"].append(case)
        print(json.dumps(case), flush=True)

    parity = all(c["tokens_equal"] and c["history_equal"] and c["rng_equal"]
                 and c["remaining_masks"] == 0 for c in report["cases"])
    if parity:
        torch.manual_seed(0)
        scorer = PromptUtilityStudent(StudentConfig(
            layer_count=backend.n_layers, hidden_dim=backend.hidden_dim,
            proj_dim=8, mlp_dim=16)).to(model.device).eval()
        calls = []
        hooks = [layer.register_forward_hook(
            lambda module, inputs, output: calls.append(bool(torch.isfinite(output).all())))
                 for layer in scorer.layers.values()]
        model.config.keep_ratio = 0.1
        actual = backend.generate(model, prompt, gen_length=32, steps=16, cache_scorer=scorer)
        for hook in hooks:
            hook.remove()
        report["scorer_execution"] = {
            "weights": "random, execution check only", "keep_ratio": 0.1,
            "layer_calls": len(calls), "finite_scores": all(calls),
            "remaining_masks": int((actual[:, prompt.shape[1]:] == backend.mask_id).sum())}
        parity = len(calls) == backend.n_layers and all(calls) and (
            report["scorer_execution"]["remaining_masks"] == 0)

    try:
        require_matching_decoding(None, DreamDecoding().metadata(), "legacy label")
    except ValueError:
        report["legacy_labels_rejected"] = True
    else:
        report["legacy_labels_rejected"] = False
        parity = False
    report["passed"] = parity
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(report, indent=2) + "\n")
    print(f"passed={parity} report={destination}", flush=True)
    return 0 if parity else 1


if __name__ == "__main__":
    raise SystemExit(main())
