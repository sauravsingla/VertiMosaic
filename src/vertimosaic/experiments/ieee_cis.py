from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass(frozen=True)
class IEEECISPrepared:
    transaction: pd.DataFrame
    identity: pd.DataFrame
    target: pd.Series


def prepare_ieee_cis(transaction_path: Path, identity_path: Path) -> IEEECISPrepared:
    """Prepare authorized local IEEE-CIS files without redistributing them."""
    transaction = pd.read_csv(transaction_path)
    identity = pd.read_csv(identity_path)
    required_transaction = {"TransactionID", "isFraud"}
    if not required_transaction.issubset(transaction.columns):
        raise ValueError("transaction file must contain TransactionID and isFraud")
    if "TransactionID" not in identity.columns:
        raise ValueError("identity file must contain TransactionID")
    target = pd.to_numeric(transaction.pop("isFraud"), errors="raise").astype(float)
    transaction_ids = transaction["TransactionID"].copy()
    transaction = transaction.set_index("TransactionID", drop=True)
    identity = identity.set_index("TransactionID", drop=True)
    common = transaction.index.intersection(identity.index)
    target_by_id = pd.Series(target.to_numpy(), index=transaction_ids.to_numpy())
    return IEEECISPrepared(
        transaction=transaction.loc[common].copy(),
        identity=identity.loc[common].copy(),
        target=target_by_id.loc[common].reset_index(drop=True),
    )
