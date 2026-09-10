"""The Sparse-dLLM Dream decoding contract, shared by training and evaluation."""

from dataclasses import asdict, dataclass
import hashlib
import math


DECODER_VERSION = "sparse_dllm_dream_v1"


@dataclass(frozen=True)
class DreamDecoding:
    alg: str = "entropy"
    temperature: float = 0.2
    top_p: float = 0.95
    steps: int = 256
    eps: float = 1e-3
    top_k: int | None = None
    alg_temp: float | None = None

    def __post_init__(self):
        if self.alg not in ("entropy", "maskgit_plus", "topk_margin"):
            raise ValueError(f"unsupported Dream alg: {self.alg}")
        if not math.isfinite(self.temperature) or self.temperature < 0:
            raise ValueError("Dream temperature must be finite and non-negative")
        if not 0 < self.top_p <= 1 or not 0 < self.eps <= 1:
            raise ValueError("Dream top_p and eps must be in (0, 1]")
        if self.steps < 2 or (self.top_k is not None and self.top_k < 1):
            raise ValueError("Dream steps must be >= 2 and top_k must be positive")
        if self.alg_temp is not None and (
            not math.isfinite(self.alg_temp) or self.alg_temp < 0
        ):
            raise ValueError("Dream alg_temp must be finite and non-negative")

    def steps_for_length(self, gen_length):
        return min(self.steps, gen_length)

    def metadata(self):
        return {"implementation": DECODER_VERSION, **asdict(self)}

    def generation_kwargs(self, gen_length):
        return {**asdict(self), "steps": self.steps_for_length(gen_length)}

    @classmethod
    def from_args(cls, args):
        return cls(**{key: getattr(args, f"dream_{key}") for key in asdict(cls())})


def add_dream_arguments(parser):
    defaults = DreamDecoding()
    for name, kind in (("alg", str), ("temperature", float), ("top_p", float),
                       ("steps", int), ("eps", float), ("top_k", int),
                       ("alg_temp", float)):
        parser.add_argument(f"--dream-{name.replace('_', '-')}", type=kind,
                            default=getattr(defaults, name))


def require_matching_decoding(saved, expected, source):
    if saved != expected:
        raise ValueError(
            f"{source}: Dream decoding metadata is missing or mismatched. "
            "Use labels/checkpoints from this decoding configuration in a new "
            f"artifact directory. expected={expected}, found={saved}"
        )


def sample_seed(seed, key):
    """Make sampling independent of which preceding examples were resumed."""
    digest = hashlib.sha256(f"{seed}:{key}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**63 - 1)
