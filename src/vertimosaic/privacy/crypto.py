# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import hashlib
import hmac
import secrets
from dataclasses import dataclass
from importlib import import_module
from typing import Any

import numpy as np

_FIELD_PRIME = 2**61 - 1


def _as_int_vector(value: np.ndarray) -> np.ndarray:
    array = np.asarray(value)
    if np.issubdtype(array.dtype, np.integer):
        return array.astype(object, copy=False)
    if array.dtype == object and all(isinstance(item, (int, np.integer)) for item in array.flat):
        return array.astype(object, copy=False)
    raise TypeError("cryptographic reference backends require integer vectors")


def _signed_from_field(value: np.ndarray, prime: int) -> np.ndarray:
    residue = np.asarray(value, dtype=object)
    half = prime // 2
    signed = np.where(residue > half, residue - prime, residue)
    return np.asarray(signed, dtype=object)


def _pairwise_mask(
    secret: bytes,
    *,
    round_id: str,
    shape: tuple[int, ...],
    prime: int,
) -> np.ndarray:
    count = int(np.prod(shape, dtype=np.int64))
    values: list[int] = []
    for index in range(count):
        payload = f"{round_id}:{index}".encode()
        digest = hmac.new(secret, payload, hashlib.sha256).digest()
        values.append(int.from_bytes(digest, "big") % prime)
    return np.asarray(values, dtype=object).reshape(shape)


@dataclass(frozen=True)
class PairwiseMaskSecureAggregation:
    """Reference pairwise-mask secure sum for integer vectors.

    Every pair of parties must receive the same out-of-band secret without exposing
    it to the aggregator. Pairwise masks cancel in the aggregate. This models the
    core honest-but-curious secure-aggregation idea, but intentionally does not
    provide dropout recovery, malicious-party security, or collusion resistance.
    """

    prime: int = _FIELD_PRIME

    def __post_init__(self) -> None:
        if self.prime <= 3:
            raise ValueError("prime must be greater than 3")

    @staticmethod
    def pair_key(left: str, right: str) -> tuple[str, str]:
        if left == right:
            raise ValueError("a pairwise secret requires two distinct parties")
        return (left, right) if left < right else (right, left)

    def mask_update(
        self,
        *,
        party_id: str,
        update: np.ndarray,
        party_ids: tuple[str, ...],
        pairwise_secrets: dict[tuple[str, str], bytes],
        round_id: str,
    ) -> np.ndarray:
        if party_id not in party_ids:
            raise ValueError("party_id must be included in party_ids")
        if len(set(party_ids)) != len(party_ids):
            raise ValueError("party_ids must be unique")
        encoded = np.mod(_as_int_vector(update), self.prime)
        masked = encoded.copy()
        for peer in party_ids:
            if peer == party_id:
                continue
            key = self.pair_key(party_id, peer)
            secret = pairwise_secrets.get(key)
            if secret is None or len(secret) < 16:
                raise ValueError(f"missing or weak pairwise secret for {key}")
            mask = _pairwise_mask(
                secret,
                round_id=round_id,
                shape=tuple(encoded.shape),
                prime=self.prime,
            )
            if party_id < peer:
                masked = np.mod(masked + mask, self.prime)
            else:
                masked = np.mod(masked - mask, self.prime)
        return np.asarray(masked, dtype=object)

    def aggregate(self, masked_updates: list[np.ndarray]) -> np.ndarray:
        if not masked_updates:
            raise ValueError("at least one masked update is required")
        shape = np.asarray(masked_updates[0]).shape
        total = np.zeros(shape, dtype=object)
        for update in masked_updates:
            array = _as_int_vector(update)
            if array.shape != shape:
                raise ValueError("all masked updates must have equal shape")
            total = np.mod(total + array, self.prime)
        return _signed_from_field(total, self.prime)

    @property
    def security_scope(self) -> str:
        return (
            "honest-but-curious aggregator reference; pairwise secrets stay out of band; "
            "no dropout recovery, malicious-party security, or collusion resistance"
        )


@dataclass(frozen=True)
class AdditiveSecretSharingSum:
    """Reference additive secret-sharing sum over a prime field.

    This is a narrow MPC primitive for integer-vector summation, not a general MPC
    runtime. Shares are sampled with Python's cryptographic ``secrets`` module.
    """

    share_count: int
    prime: int = _FIELD_PRIME

    def __post_init__(self) -> None:
        if self.share_count < 2:
            raise ValueError("share_count must be at least 2")
        if self.prime <= 3:
            raise ValueError("prime must be greater than 3")

    def split(self, value: np.ndarray) -> list[np.ndarray]:
        encoded = np.mod(_as_int_vector(value), self.prime)
        shares: list[np.ndarray] = []
        running = np.zeros(encoded.shape, dtype=object)
        for _ in range(self.share_count - 1):
            random_values = [secrets.randbelow(self.prime) for _ in range(encoded.size)]
            share = np.asarray(random_values, dtype=object).reshape(encoded.shape)
            shares.append(share)
            running = np.mod(running + share, self.prime)
        shares.append(np.mod(encoded - running, self.prime))
        return shares

    def reconstruct(self, shares: list[np.ndarray]) -> np.ndarray:
        if len(shares) != self.share_count:
            raise ValueError("incorrect number of shares")
        shape = np.asarray(shares[0]).shape
        total = np.zeros(shape, dtype=object)
        for share in shares:
            array = _as_int_vector(share)
            if array.shape != shape:
                raise ValueError("all shares must have equal shape")
            total = np.mod(total + array, self.prime)
        return _signed_from_field(total, self.prime)

    @property
    def security_scope(self) -> str:
        return (
            "additive secret sharing for integer-vector sums only; not a general MPC "
            "runtime and no malicious-party verification"
        )


@dataclass(frozen=True)
class OpenMinedPSIBackend:
    """Optional adapter for OpenMined's ECDH-based PSI implementation."""

    false_positive_rate: float = 1e-9

    def __post_init__(self) -> None:
        if not 0.0 < self.false_positive_rate < 1.0:
            raise ValueError("false_positive_rate must be in (0, 1)")

    @staticmethod
    def _module() -> Any:
        candidates = ("openmined_psi", "private_set_intersection.python")
        for name in candidates:
            try:
                return import_module(name)
            except ImportError:
                continue
        raise RuntimeError(
            "OpenMined PSI is optional; install VertiMosaic with the 'privacy-crypto' extra"
        )

    def intersection(self, client_ids: list[str], server_ids: list[str]) -> list[str]:
        module = self._module()
        client = module.client.CreateWithNewKey(True)
        server = module.server.CreateWithNewKey(True)
        data_structure = getattr(getattr(module, "DataStructure", object), "RAW", None)
        if data_structure is None:
            setup = server.CreateSetupMessage(
                self.false_positive_rate,
                len(client_ids),
                server_ids,
            )
        else:
            setup = server.CreateSetupMessage(
                self.false_positive_rate,
                len(client_ids),
                server_ids,
                data_structure,
            )
        request = client.CreateRequest(client_ids)
        response = server.ProcessRequest(request)
        indices = client.GetIntersection(setup, response)
        return [client_ids[int(index)] for index in indices]

    @property
    def security_scope(self) -> str:
        return (
            "optional OpenMined ECDH PSI adapter; PSI protects set-intersection protocol "
            "inputs subject to the upstream implementation's threat model, not VFL "
            "training messages"
        )


@dataclass(frozen=True)
class PaillierHomomorphicSum:
    """Optional Paillier adapter for bounded signed integer-vector summation."""

    key_bits: int = 1024

    def __post_init__(self) -> None:
        if self.key_bits < 512:
            raise ValueError("key_bits must be at least 512 for research use")

    @staticmethod
    def _module() -> Any:
        try:
            return import_module("pailliers")
        except ImportError as exc:
            raise RuntimeError(
                "Paillier support is optional; install VertiMosaic with the 'privacy-crypto' extra"
            ) from exc

    def sum(self, vectors: list[np.ndarray]) -> np.ndarray:
        if not vectors:
            raise ValueError("at least one vector is required")
        arrays = [_as_int_vector(vector) for vector in vectors]
        shape = arrays[0].shape
        if any(array.shape != shape for array in arrays):
            raise ValueError("all vectors must have equal shape")
        module = self._module()
        secret_key = module.secret(self.key_bits)
        public_key = module.public(secret_key)
        modulus = int(public_key[0])
        true_sum_bound = sum(int(np.max(np.abs(array.astype(object)))) for array in arrays)
        if true_sum_bound >= modulus // 3:
            raise OverflowError("plaintext sum is too large for the generated Paillier modulus")
        result: list[int] = []
        for index in range(arrays[0].size):
            encrypted = [module.encrypt(public_key, int(array.flat[index])) for array in arrays]
            total_cipher = sum(encrypted)
            residue = int(module.decrypt(secret_key, total_cipher))
            result.append(residue - modulus if residue > modulus // 2 else residue)
        return np.asarray(result, dtype=object).reshape(shape)

    @property
    def security_scope(self) -> str:
        return (
            "optional Paillier additive-homomorphic integer sum; does not by itself provide "
            "secure VFL, key management, malicious-party security, or private comparisons"
        )
