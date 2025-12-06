import pandas as pd
import numpy as np
import random
from faker import Faker

fake = Faker()
random.seed(42)
np.random.seed(42)

# ---------------- CONFIG ----------------
INPUT_CSV  = "enhanced_health_insurance_claims.csv"   # Kaggle file
OUTPUT_CSV = "payer_unsupervised_anomaly_data.csv"

TARGET_ROWS        = 3000   # choose 1000–5000
DUPLICATE_RATE     = 0.03   # 3% extra duplicate rows
MISSING_RATE       = 0.05   # 5% missing per column
FORMAT_RATE        = 0.03   # 3% rows get format errors
OUTLIER_RATE       = 0.02   # 2% rows get amount outliers
INCONSISTENCY_RATE = 0.02   # 2% rows get logic inconsistencies
# ----------------------------------------

RELEVANT_COLS = [
    "ClaimID", "PatientID", "ProviderID",
    "ClaimDate", "ServiceDate", "DOB",
    "DiagnosisCode", "ProcedureCode",
    "ClaimAmount",
    "PatientGender", "ZIP", "State",
    "ProviderSpecialty", "ProviderLocation",
    "ClaimStatus", "ClaimType"
]


# ---------- Helpers for anomaly strings ----------

def random_bad_date():
    return random.choice([
        "99/99/9999", "2025-13-40", "abcd", "12/2025/40"
    ])

def random_bad_zip():
    return random.choice(["ABCDE", "123", "123456", "12A45"])

def random_bad_state():
    return random.choice(["XX", "??", "A1"])

def random_bad_amount():
    return random.choice(["--100", "$5000", "12a3", "100O"])

def random_bad_code():
    return random.choice(["A1@@#", "XY123", "12ABC", "A.123"])


# ---------- Stage 1: Load Kaggle & add extra columns ----------

def load_and_prepare():
    df = pd.read_csv(INPUT_CSV)

    base = pd.DataFrame()
    base["ClaimID"]           = df["ClaimID"]
    base["PatientID"]         = df["PatientID"]
    base["ProviderID"]        = df["ProviderID"]
    base["ClaimAmount"]       = df["ClaimAmount"]
    base["ClaimDate"]         = df["ClaimDate"]
    base["DiagnosisCode"]     = df["DiagnosisCode"]
    base["ProcedureCode"]     = df["ProcedureCode"]
    # base["PatientAge"]        = df["PatientAge"]
    base["PatientGender"]     = df["PatientGender"]
    base["ProviderSpecialty"] = df["ProviderSpecialty"]
    base["ProviderLocation"]  = df["ProviderLocation"]
    base["ClaimStatus"]       = df["ClaimStatus"]
    base["ClaimType"]         = df["ClaimType"]

    # add new synthetic columns (no strict consistency enforcement)
    base["DOB"] = [
        fake.date_of_birth(minimum_age=10, maximum_age=90).strftime("%Y-%m-%d")
        for _ in range(len(base))
    ]
    base["ServiceDate"] = [
        fake.date_between(start_date="-2y", end_date="today").strftime("%Y-%m-%d")
        for _ in range(len(base))
    ]
    base["ZIP"] = [fake.postcode().replace(" ", "")[:5] for _ in range(len(base))]
    base["State"] = [fake.state_abbr() for _ in range(len(base))]

    base = base[RELEVANT_COLS]
    return base


# ---------- Stage 2: Expand to target rows (simple sampling + jitter) ----------

def expand_to_target(df):
    if len(df) >= TARGET_ROWS:
        return df.sample(TARGET_ROWS, random_state=42).reset_index(drop=True)

    need = TARGET_ROWS - len(df)
    rows = df.to_dict("records")

    for _ in range(need):
        r = df.sample(1).iloc[0].copy()

        # slight jitter in claim amount
        try:
            amt = float(r["ClaimAmount"])
            r["ClaimAmount"] = round(max(0, amt + np.random.normal(0, 0.05 * max(amt, 1))), 2)
        except:
            r["ClaimAmount"] = round(np.random.uniform(50, 5000), 2)

        # small random shift in dates
        for col in ["ClaimDate", "ServiceDate", "DOB"]:
            try:
                d = pd.to_datetime(r[col], errors="coerce")
                if pd.notna(d):
                    r[col] = (d + pd.Timedelta(days=np.random.randint(-10, 11))).strftime("%Y-%m-%d")
            except:
                pass

        # give it a new ClaimID so duplicates only come from anomaly injection
        r["ClaimID"] = fake.uuid4()
        rows.append(r)

    return pd.DataFrame(rows)[RELEVANT_COLS]


# ---------- Stage 3: Inject anomalies (unsupervised) ----------

def add_duplicates(df):
    n_extra = int(len(df) * DUPLICATE_RATE)
    if n_extra <= 0: return df
    dup = df.sample(n_extra, random_state=100)
    # these rows have same ClaimID etc. → true duplicates
    df2 = pd.concat([df, dup], ignore_index=True)
    return df2

def add_missing_values(df):
    for col in RELEVANT_COLS:
        n = int(len(df) * MISSING_RATE)
        if n <= 0: continue
        idx = df.sample(n, random_state=random.randint(0, 99999)).index
        df.loc[idx, col] = np.nan
    return df

def add_format_anomalies(df):
    col_to_gen = {
        "ClaimDate":     random_bad_date,
        "ServiceDate":   random_bad_date,
        "DOB":           random_bad_date,
        "ZIP":           random_bad_zip,
        "State":         random_bad_state,
        "ClaimAmount":   random_bad_amount,
        "DiagnosisCode": random_bad_code,
        "ProcedureCode": random_bad_code
    }

    n = int(len(df) * FORMAT_RATE)
    if n <= 0: return df

    idx = df.sample(n, random_state=200).index
    for i in idx:
        col = random.choice(list(col_to_gen.keys()))
        df.at[i, col] = col_to_gen[col]()
    return df

def add_outliers(df):
    n = int(len(df) * OUTLIER_RATE)
    if n <= 0: return df
    idx = df.sample(n, random_state=300).index
    for i in idx:
        if random.random() < 0.5:
            # huge positive
            df.at[i, "ClaimAmount"] = round(np.random.uniform(200000, 5000000), 2)
        else:
            # negative amount
            df.at[i, "ClaimAmount"] = -round(np.random.uniform(10, 2000), 2)
    return df

def add_inconsistencies(df):
    """
    Logical inconsistencies like:
    - DOB after ClaimDate (age < 0)
    - PatientAge < 18 randomly
    - ServiceDate after ClaimDate 
    """
    n = int(len(df) * INCONSISTENCY_RATE)
    if n <= 0: return df

    idx = df.sample(n, random_state=400).index
    for i in idx:
        # randomly choose type of inconsistency
        t = random.choice(["dob_after_claim", "underage", "service_after_claim"])
        if t == "dob_after_claim":
            cd = pd.to_datetime(df.at[i, "ClaimDate"], errors="coerce")
            if pd.notna(cd):
                df.at[i, "DOB"] = (cd + pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        # elif t == "underage":
        #     df.at[i, "PatientAge"] = random.randint(1, 16)
        elif t == "service_after_claim":
            cd = pd.to_datetime(df.at[i, "ClaimDate"], errors="coerce")
            if pd.notna(cd):
                df.at[i, "ServiceDate"] = (cd + pd.Timedelta(days=random.randint(1, 60))).strftime("%Y-%m-%d")
    return df


# ---------- MAIN ----------

def main():
    df = load_and_prepare()
    df = expand_to_target(df)

    # inject anomalies – unsupervised, so no labels
    df = add_duplicates(df)
    df = add_missing_values(df)
    df = add_format_anomalies(df)
    df = add_outliers(df)
    df = add_inconsistencies(df)

    # shuffle rows
    df = df.sample(frac=1, random_state=999).reset_index(drop=True)

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved unsupervised anomaly dataset to: {OUTPUT_CSV}")
    print(f"Final rows: {len(df)}")

if __name__ == "__main__":
    main()
