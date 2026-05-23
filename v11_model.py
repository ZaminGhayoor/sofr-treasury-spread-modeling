import pandas as pd
import numpy as np
from scipy.interpolate import PchipInterpolator

curve = pd.read_csv("treasury_curve_2023_05_15.csv")
curve["years"] = pd.to_numeric(curve["years"], errors="coerce")
curve["yield"] = pd.to_numeric(curve["yield"], errors="coerce")
curve = curve.dropna().sort_values("years")
curve_fn = PchipInterpolator(curve["years"].values, curve["yield"].values)

df = pd.read_csv("floater_merged.csv", low_memory=False)
df["Price"] = pd.to_numeric(df["Price"], errors="coerce")
df["Coupon_trade"] = pd.to_numeric(df["Coupon_trade"], errors="coerce")
df["ReportedYield"] = pd.to_numeric(df["ReportedYield"], errors="coerce")
df["TradeDate"] = pd.to_datetime(df["TradeDate"], errors="coerce")
df["Maturity"] = pd.to_datetime(df["Maturity"], errors="coerce")
if "IssueName" not in df.columns:
    df["IssueName"] = ""
else:
    df["IssueName"] = df["IssueName"].fillna("").astype(str)

df = df[
    df["Price"].notna() &
    df["Coupon_trade"].notna() &
    df["ReportedYield"].notna() &
    df["TradeDate"].notna() &
    df["Maturity"].notna()
].copy()

print("Rows used for evaluation:", len(df))

df["years_to_maturity"] = (df["Maturity"] - df["TradeDate"]).dt.days / 365.25
df = df[df["years_to_maturity"] > 0].copy()

min_curve = curve["years"].min()
max_curve = curve["years"].max()
df["years_clipped"] = df["years_to_maturity"].clip(lower=min_curve, upper=max_curve)

df["BenchmarkRate"] = curve_fn(df["years_clipped"].values)
df["Spread"] = df["ReportedYield"] - df["BenchmarkRate"]
df["Discount"] = 100 - df["Price"]
df["Issuer"] = df["IssueName"].str.split(",").str[0].str.strip()

issuer_sum = df.groupby("Issuer")["Spread"].transform("sum")
issuer_count = df.groupby("Issuer")["Spread"].transform("count")
global_mean = df["Spread"].mean()

df["IssuerEffect_LOO"] = np.where(
    issuer_count > 1,
    (issuer_sum - df["Spread"]) / (issuer_count - 1),
    global_mean
)

df["SpreadAfterIssuer"] = df["Spread"] - df["IssuerEffect_LOO"]

bond_sum = df.groupby("CUSIP")["SpreadAfterIssuer"].transform("sum")
bond_count = df.groupby("CUSIP")["SpreadAfterIssuer"].transform("count")

df["BondEffect_LOO"] = np.where(
    bond_count > 1,
    (bond_sum - df["SpreadAfterIssuer"]) / (bond_count - 1),
    0.0
)

df["ResidualSpread"] = df["Spread"] - df["IssuerEffect_LOO"] - df["BondEffect_LOO"]

X0 = np.ones(len(df))
X1 = df["Coupon_trade"].values
X2 = df["Discount"].values
X3 = df["years_to_maturity"].values

X = np.vstack([X0, X1, X2, X3]).T
y = df["ResidualSpread"].values

beta = np.linalg.pinv(X) @ y
a, b1, b2, b3 = beta

print("\nGlobal spread mean:", round(global_mean, 6))
print("\nResidual model coefficients:")
print(f"  intercept = {a:.6f}")
print(f"  coupon    = {b1:.6f}")
print(f"  discount  = {b2:.6f}")
print(f"  years     = {b3:.6f}")

df["ResidualPred"] = a + b1 * df["Coupon_trade"] + b2 * df["Discount"] + b3 * df["years_to_maturity"]

df["ComputedYield"] = df["BenchmarkRate"] + df["IssuerEffect_LOO"] + df["BondEffect_LOO"] + df["ResidualPred"]
df["YieldErrorBps"] = (df["ComputedYield"] - df["ReportedYield"]) * 100
df["AbsErrorBps"] = df["YieldErrorBps"].abs()

issuer_n = df.groupby("Issuer")["Spread"].transform("count")
bond_n = df.groupby("CUSIP")["Spread"].transform("count")
df["issuer_n"] = issuer_n
df["bond_n"] = bond_n

print("\n===== V11 MODEL REPORT =====")
print("Mean abs error (bps):", df["AbsErrorBps"].mean())
print("Median abs error (bps):", df["AbsErrorBps"].median())
print("Min error (bps):", df["YieldErrorBps"].min())
print("Max error (bps):", df["YieldErrorBps"].max())

print("\nBest 5:")
print(
    df.nsmallest(5, "AbsErrorBps")[
        ["CUSIP", "IssueName", "ReportedYield", "ComputedYield", "YieldErrorBps", "issuer_n", "bond_n"]
    ].to_string(index=False)
)

print("\nWorst 5:")
print(
    df.nlargest(5, "AbsErrorBps")[
        ["CUSIP", "IssueName", "ReportedYield", "ComputedYield", "YieldErrorBps", "issuer_n", "bond_n"]
    ].to_string(index=False)
)

df.to_csv("v11_results.csv", index=False)
print("\nSaved: v11_results.csv")
