from fib_train_verifier import verify_symbol_fib_train
import pandas as pd

verify_symbol_fib_train(
    pair="DYMUSDT",
    interval="3m",                   # or "3m"
    train_start=pd.Timestamp("2025-11-06 00:00:00"),
    train_end=pd.Timestamp("2025-11-08 23:59:00"),
    initial_capital=10_000.0,
    trade_size=1_000.0,
    verbose=True,
)