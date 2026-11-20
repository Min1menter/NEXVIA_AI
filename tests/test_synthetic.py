from fraudshield.synthetic import generate


def test_schema_and_imbalance():
    df = generate(n_users=500, n_txn=20_000, fraud_rate=0.008, n_rings=5, seed=1)
    assert 0.003 < df.is_fraud.mean() < 0.02              # heavy imbalance (<2%)
    assert df.timestamp.is_monotonic_increasing and df.timestamp.is_unique
    assert df.txn_id.is_unique
    assert {"device_id", "card_id", "country", "is_fraud", "ring_id"} <= set(df.columns)


def test_rings_share_devices():
    df = generate(n_users=500, n_txn=20_000, n_rings=5, seed=1)
    ring = df[df.ring_id >= 0]
    users_per_device = ring.groupby("device_id").user_id.nunique()
    assert users_per_device.max() >= 5                    # ring members share devices => graph signal
