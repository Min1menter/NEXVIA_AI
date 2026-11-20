"""Feast definitions. `user_state` is the per-user state used by fraudshield.features (kept fresh via PushSource);
`graph_scores` holds the GNN output per user (batch, materialized after each graph training run)."""
import os
from datetime import timedelta
from feast import Entity, FeatureView, Field, FileSource, PushSource
from feast.types import Float32, Float64, Int64, String

DATA = os.getenv("DATA_DIR", "../data")

user = Entity(name="user", join_keys=["user_id"])

user_state_batch = FileSource(name="user_state_batch", path=f"{DATA}/state_snapshots.parquet",
                              timestamp_field="event_timestamp")
user_state_push = PushSource(name="user_state_push", batch_source=user_state_batch)

user_state = FeatureView(
    name="user_state", entities=[user], ttl=timedelta(days=3650), online=True, source=user_state_push,
    schema=[Field(name="n_txn", dtype=Int64), Field(name="mean_amt", dtype=Float64), Field(name="m2_amt", dtype=Float64),
            Field(name="ew_cnt_short", dtype=Float64), Field(name="ew_cnt_long", dtype=Float64),
            Field(name="last_ts", dtype=Float64), Field(name="last_device", dtype=String),
            Field(name="last_country", dtype=String)])

graph_scores_src = FileSource(name="graph_scores_src", path=f"{DATA}/graph_scores.parquet", timestamp_field="event_timestamp")
graph_scores = FeatureView(
    name="graph_scores", entities=[user], ttl=timedelta(days=3650), online=True, source=graph_scores_src,
    schema=[Field(name="graph_score", dtype=Float32), Field(name="cluster_size", dtype=Int64)])
