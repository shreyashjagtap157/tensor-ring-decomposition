from __future__ import annotations

import enum
import json
import logging
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class ModelFamily(enum.Enum):
    BERT = "bert"
    ROBERTA = "roberta"
    GPT2 = "gpt2"
    LLAMA = "llama"
    MISTRAL = "mistral"
    FALCON = "falcon"
    OPT = "opt"
    T5 = "t5"
    BART = "bart"
    DEBERTA = "deberta"
    ELECTRA = "electra"
    XLNET = "xlnet"
    ALBERT = "albert"
    MPNET = "mpnet"
    DISTILBERT = "distilbert"
    CAMEMBERT = "camembert"
    BLOOM = "bloom"
    CODELLAMA = "codellama"
    QWEN2 = "qwen2"
    GEMMA = "gemma"
    PHI = "phi"
    MIXTRAL = "mixtral"
    CUSTOM = "custom"


@dataclass
class ModelProfile:
    """Compression profile for a specific model."""

    name: str
    family: ModelFamily
    vocab_size: int
    embedding_dim: int
    recommended_ranks: List[int] = field(default_factory=lambda: [4, 8, 16, 32, 64])
    default_rank: int = 8
    ring_components: int = 4
    notes: str = ""
    max_seq_len: Optional[int] = None
    padding_token_id: Optional[int] = None
    has_position_embeddings: bool = True
    has_token_type_embeddings: bool = False

    @property
    def dense_params(self) -> int:
        return self.vocab_size * self.embedding_dim

    def compression_at_rank(self, rank: int) -> float:
        from ..core.factorization import compute_ring_structure
        struct = compute_ring_structure(
            self.vocab_size, self.embedding_dim, self.ring_components, rank
        )
        total = 0
        for i in range(struct.n_vocab_cores):
            total += struct.vocab_factor_sizes[i] * rank * rank
        for i in range(struct.n_emb_cores):
            total += struct.emb_factor_sizes[i] * rank * rank
        return self.dense_params / total

    def rank_for_compression(self, target_ratio: float) -> int:
        from ..core.embedding import TensorRingEmbedding
        return TensorRingEmbedding.optimal_rank(
            self.vocab_size, self.embedding_dim,
            self.ring_components, target_compression=target_ratio,
        )

    def params_at_rank(self, rank: int) -> int:
        from ..core.factorization import compute_ring_structure
        struct = compute_ring_structure(
            self.vocab_size, self.embedding_dim, self.ring_components, rank
        )
        total = 0
        for i in range(struct.n_vocab_cores):
            total += struct.vocab_factor_sizes[i] * rank * rank
        for i in range(struct.n_emb_cores):
            total += struct.emb_factor_sizes[i] * rank * rank
        return total

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["family"] = self.family.value
        return d


class ModelRegistry:
    """Registry of known model profiles with lookup capabilities."""

    _profiles: Dict[str, ModelProfile] = {}

    @classmethod
    def register(cls, profile: ModelProfile) -> None:
        key = profile.name.lower()
        if key in cls._profiles:
            logger.warning(f"Overwriting existing profile for '{profile.name}'")
        cls._profiles[key] = profile
        logger.debug(f"Registered model: {profile.name}")

    @classmethod
    def get(cls, name: str) -> Optional[ModelProfile]:
        return cls._profiles.get(name.lower())

    @classmethod
    def lookup(cls, vocab_size: int, embedding_dim: int) -> List[ModelProfile]:
        matches = []
        for profile in cls._profiles.values():
            if profile.vocab_size == vocab_size and profile.embedding_dim == embedding_dim:
                matches.append(profile)
        return matches

    @classmethod
    def list_by_family(cls, family: ModelFamily) -> List[ModelProfile]:
        return [p for p in cls._profiles.values() if p.family == family]

    @classmethod
    def list_all(cls) -> List[ModelProfile]:
        return list(cls._profiles.values())

    @classmethod
    def summary(cls) -> str:
        lines = ["Model Registry Summary:"]
        for prof in sorted(cls._profiles.values(), key=lambda p: p.name):
            c4 = prof.compression_at_rank(4)
            c32 = prof.compression_at_rank(32)
            lines.append(
                f"  {prof.name:25s}  V={prof.vocab_size:<6}  D={prof.embedding_dim:<4}  "
                f"{c4:>5.0f}x@R4  {c32:>5.0f}x@R32  [{prof.family.value}]"
            )
        return "\n".join(lines)

    @classmethod
    def from_json(cls, path: str) -> None:
        data = json.loads(Path(path).read_text())
        for item in data:
            item["family"] = ModelFamily(item["family"])
            cls.register(ModelProfile(**item))

    @classmethod
    def to_json(cls, path: str) -> None:
        data = [p.to_dict() for p in cls._profiles.values()]
        Path(path).write_text(json.dumps(data, indent=2))
