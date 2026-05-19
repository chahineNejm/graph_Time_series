"""Data loading utilities — uses gift_eval.data.Dataset (GiftEval benchmark)."""

import numpy as np
from gift_eval.data import Dataset


def load_gifteval_config(
    dataset_name: str,
    *,
    term: str = "short",
    max_samples: int = 25,
    to_univariate: bool = True,
    storage_env_var: str = "GIFT_EVAL",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a GiftEval config and return (history, future) arrays.

    Uses gift_eval.data.Dataset to load training + validation data.
    For each series the validation target is split into context (= training)
    and future (= the last prediction_length values).

    Returns:
        H : (n_samples, context_length) — history / context
        F : (n_samples, horizon)        — ground-truth future
    """
    dataset = Dataset(
        name=dataset_name,
        term=term,
        to_univariate=to_univariate,
        storage_env_var=storage_env_var,
    )

    histories = []
    futures = []

    for i, (train_entry, val_entry) in enumerate(
        zip(dataset.training_dataset, dataset.validation_dataset)
    ):
        if i >= max_samples:
            break

        train_ts = np.asarray(train_entry["target"], dtype=np.float32).ravel()
        val_ts = np.asarray(val_entry["target"], dtype=np.float32).ravel()

        # The validation target = context || future
        # context length = len(train), future = everything after
        future = val_ts[len(train_ts):]
        if len(future) == 0 or len(train_ts) < 10:
            continue

        histories.append(train_ts)
        futures.append(future)

    if not histories:
        raise ValueError(f"No usable series found in {dataset_name}/{term}")

    # Truncate to common lengths
    min_hist = min(len(x) for x in histories)
    min_fut = min(len(x) for x in futures)

    H = np.array([x[-min_hist:] for x in histories], dtype=np.float32)
    F = np.array([x[:min_fut] for x in futures], dtype=np.float32)

    print(f"  {dataset_name}/{term}: {H.shape[0]} samples, "
          f"history={H.shape[1]}, horizon={F.shape[1]}")
    return H, F


def load_multiple_configs(
    config_names: list[str],
    *,
    term: str = "short",
    max_samples: int = 25,
    to_univariate: bool = True,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Load several GiftEval configs, skip any that fail."""
    from tqdm.auto import tqdm

    data = {}
    for cfg in tqdm(config_names, desc="Loading configs"):
        try:
            data[cfg] = load_gifteval_config(
                cfg, term=term, max_samples=max_samples,
                to_univariate=to_univariate,
            )
        except Exception as e:
            print(f"  SKIP {cfg}: {e}")
    print(f"\nLoaded {len(data)} configs")
    return data
