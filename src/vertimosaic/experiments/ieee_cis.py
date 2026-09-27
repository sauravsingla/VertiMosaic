from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IEEECISPrepared:
    transaction: pd.DataFrame
    identity: pd.DataFrame
    target: pd.Series
    entity_alignment_seconds: float = 0.0
    transaction_source_rows: int = 0
    identity_source_rows: int = 0


def prepare_ieee_cis(transaction_path: Path, identity_path: Path) -> IEEECISPrepared:
    """Prepare authorized local IEEE-CIS files without redistributing them."""
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    transaction_source_rows = len(transaction)
    identity_source_rows = len(identity)
    required_transaction = {"TransactionID", "isFraud"}
    if not required_transaction.issubset(transaction.columns):
        raise ValueError("transaction file must contain TransactionID and isFraud")
    if "TransactionID" not in identity.columns:
        raise ValueError("identity file must contain TransactionID")
    target = pd.to_numeric(transaction.pop("isFraud"), errors="raise").astype(float)
    transaction_ids = transaction["TransactionID"].copy()
    alignment_start = time.perf_counter()
    transaction = transaction.set_index("TransactionID", drop=True)
    identity = identity.set_index("TransactionID", drop=True)
    common = transaction.index.intersection(identity.index)
    target_by_id = pd.Series(target.to_numpy(), index=transaction_ids.to_numpy())
    transaction_aligned = transaction.loc[common].copy()
    identity_aligned = identity.loc[common].copy()
    target_aligned = target_by_id.loc[common].reset_index(drop=True)
    entity_alignment_seconds = time.perf_counter() - alignment_start
    return IEEECISPrepared(
        transaction=transaction_aligned,
        identity=identity_aligned,
        target=target_aligned,
        entity_alignment_seconds=entity_alignment_seconds,
        transaction_source_rows=transaction_source_rows,
        identity_source_rows=identity_source_rows,
    )
