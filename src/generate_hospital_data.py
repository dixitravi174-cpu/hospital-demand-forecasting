import os
import numpy as np
import pandas as pd

def generate_hospital_data(seed: int = 42, output_dir: str = "data"):
    np.random.seed(seed)
    os.makedirs(output_dir, exist_ok=True)

    N_ROWS = 500_000
    hospitals = ["Pune_Hosp_Ruby", "Pune_Hosp_KEM", "Pune_Hosp_Jehangir", "Pune_Hosp_Sahyadri"]
    departments = ["General OPD", "Emergency / Trauma", "Pediatrics (Dengue/Viral)", "Internal Medicine"]

    hospital_col = np.random.choice(hospitals, size=N_ROWS, p=[0.30, 0.25, 0.25, 0.20])
    dept_col = np.random.choice(departments, size=N_ROWS, p=[0.40, 0.20, 0.20, 0.20])
    weeks_hist = np.random.choice(np.arange(0, 104), size=N_ROWS, p=np.ones(104)/104)

    diagnoses = [
        "Dengue fever suspected with high fever and platelet drop",
        "Acute viral gastroenteritis and dehydration",
        "Routine diabetes follow-up and prescription refill",
        "Hypertension routine checkup",
        "Severe dengue hemorrhagic fever admission",
        "Upper respiratory tract infection / viral flu",
        "Orthopedic joint pain and physiotherapy consult",
        "Pediatric viral fever with cough"
    ]
    diagnosis_col = np.random.choice(diagnoses, size=N_ROWS, p=[0.18, 0.15, 0.15, 0.12, 0.08, 0.15, 0.09, 0.08])
    
    opd_visits = np.random.poisson(lam=4.5, size=N_ROWS) + 1
    admissions = np.where(dept_col == "Emergency / Trauma", np.random.poisson(1.8, size=N_ROWS), np.random.poisson(0.6, size=N_ROWS))
    lab_tests = opd_visits * np.random.randint(1, 4, size=N_ROWS)
    billing_inr = (opd_visits * 850.0) + (admissions * 14500.0) + (lab_tests * 450.0)

    df_hosp = pd.DataFrame({
        "record_id": np.arange(1, N_ROWS + 1),
        "hospital_name": hospital_col,
        "department": dept_col,
        "week_idx": weeks_hist,
        "free_text_diagnosis": diagnosis_col,
        "opd_visits": opd_visits,
        "admissions": admissions,
        "lab_tests": lab_tests,
        "billing_inr": np.round(billing_inr, 2)
    })
    
    df_hosp.to_csv(os.path.join(output_dir, "hospital_records_500k.csv"), index=False)
    print(f"Successfully generated {len(df_hosp):,} hospital records in {output_dir}/")

if __name__ == "__main__":
    generate_hospital_data()
