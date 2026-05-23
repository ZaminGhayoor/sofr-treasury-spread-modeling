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

issuer_mean = df.groupby("Issuer")["Spread"].mean().to_dict()
df["IssuerEffect"] = df["Issuer"].map(issuer_mean)

df["SpreadAfterIssuer"] = df["Spread"] - df["IssuerEffect"]
bond_mean = df.groupby("CUSIP")["SpreadAfterIssuer"].mean().to_dict()
df["BondEffect"] = df["CUSIP"].map(bond_mean)

df["ResidualSpread"] = df["Spread"] - df["IssuerEffect"] - df["BondEffect"]

X0 = np.ones(len(df))
X1 = df["Coupon_trade"].values
X2 = df["Discount"].values
X3 = df["years_to_maturity"].values

X = np.vstack([X0, X1, X2, X3]).T
y = df["ResidualSpread"].values

beta = np.linalg.pinv(X) @ y
a, b1, b2, b3 = beta

print("\nResidual model coefficients:")
print(f"  intercept = {a:.6f}")
print(f"  coupon    = {b1:.6f}")
print(f"  discount  = {b2:.6f}")
print(f"  years     = {b3:.6f}")

df["ResidualPred"] = a + b1 * df["Coupon_trade"] + b2 * df["Discount"] + b3 * df["years_to_maturity"]

df["ComputedYield"] = df["BenchmarkRate"] + df["IssuerEffect"] + df["BondEffect"] + df["ResidualPred"]
df["YieldErrorBps"] = (df["ComputedYield"] - df["ReportedYield"]) * 100
df["AbsErrorBps"] = df["YieldErrorBps"].abs()

print("\n===== V10 MODEL REPORT =====")
print("Mean abs error (bps):", df["AbsErrorBps"].mean())
print("Median abs error (bps):", df["AbsErrorBps"].median())

df.to_csv("v10_results.csv", index=False)
print("\nSaved: v10_results.csv")
