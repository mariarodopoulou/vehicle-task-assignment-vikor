"""
Vehicle-to-Task Assignment using Group VIKOR

This project ranks vehicles for different tasks using the Group VIKOR
multi-criteria decision-making method.

Author: Maria Rodopoulou
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# =========================
# Load dataset
# =========================
df = pd.read_csv(r"C:\Users\maria\Desktop\GroupVicorTaskwithExampleset\final_cars_datasets.csv")

print(df.head())
print(df.columns)


# =========================
# Data cleaning
# =========================
criteria = ["price", "mileage", "engine_capacity", "year"]

df = df.dropna(subset=criteria)
df = df.head(500)

for c in criteria:
    df[c] = pd.to_numeric(df[c], errors="coerce")

df = df.dropna(subset=criteria)


# =========================
# Benefit / Cost criteria
# =========================
benefit_cost = {
    "price": "cost",
    "mileage": "cost",
    "engine_capacity": "benefit",
    "year": "benefit"
}


# =========================
# Task-specific weights
# =========================
tasks = {
    "City Delivery": [0.40, 0.30, 0.10, 0.20],
    "Long Distance Travel": [0.20, 0.40, 0.20, 0.20],
    "Budget Purchase": [0.60, 0.20, 0.10, 0.10],
    "Performance Driving": [0.10, 0.20, 0.50, 0.20],
    "Modern Family Car": [0.20, 0.20, 0.20, 0.40]
}


# =========================
# VIKOR function
# =========================
def run_vikor(data, criteria, weights, benefit_cost):
    weights = np.array(weights)

    matrix = data[criteria].astype(float).values

    f_star = []
    f_minus = []

    for i, criterion in enumerate(criteria):
        if benefit_cost[criterion] == "benefit":
            f_star.append(matrix[:, i].max())
            f_minus.append(matrix[:, i].min())
        else:
            f_star.append(matrix[:, i].min())
            f_minus.append(matrix[:, i].max())

    f_star = np.array(f_star)
    f_minus = np.array(f_minus)

    S = []
    R = []

    for row in matrix:
        values = []

        for j in range(len(criteria)):
            value = weights[j] * (
                (f_star[j] - row[j]) /
                (f_star[j] - f_minus[j] + 1e-9)
            )

            if benefit_cost[criteria[j]] == "cost":
                value = weights[j] * (
                    (row[j] - f_star[j]) /
                    (f_minus[j] - f_star[j] + 1e-9)
                )

            values.append(value)

        S.append(sum(values))
        R.append(max(values))

    S = np.array(S)
    R = np.array(R)

    v = 0.5

    Q = (
        v * ((S - S.min()) / (S.max() - S.min() + 1e-9))
        + (1 - v) * ((R - R.min()) / (R.max() - R.min() + 1e-9))
    )

    result = data.copy()
    result["S"] = S
    result["R"] = R
    result["Q"] = Q
    result["Rank"] = result["Q"].rank(method="min").astype(int)

    return result.sort_values("Rank")


# =========================
# Run assignment
# =========================
all_task_results = []
best_assignments = []

for task_name, weights in tasks.items():
    task_result = run_vikor(df, criteria, weights, benefit_cost)
    task_result["Task"] = task_name

    best_car = task_result.iloc[0]

    best_assignments.append({
        "Task": task_name,
        "Best Car ID": best_car.get("Unnamed: 0", best_car.name),
        "Mark": best_car["mark"],
        "Model": best_car["model"],
        "Year": best_car["year"],
        "Price": best_car["price"],
        "Mileage": best_car["mileage"],
        "Engine Capacity": best_car["engine_capacity"],
        "Q Score": best_car["Q"]
    })

    all_task_results.append(task_result)


all_results_df = pd.concat(all_task_results)
best_assignments_df = pd.DataFrame(best_assignments)


# =========================
# Export results
# =========================
output_file = "Vehicle_Task_Assignment_GroupVIKOR.xlsx"

with pd.ExcelWriter(output_file, engine="xlsxwriter") as writer:
    best_assignments_df.to_excel(writer, sheet_name="Best_Assignments", index=False)
    all_results_df.to_excel(writer, sheet_name="All_Task_Rankings", index=False)

    weights_df = pd.DataFrame(tasks, index=criteria)
    weights_df.to_excel(writer, sheet_name="Task_Weights")

    workbook = writer.book

    # Chart 1: Best assignments Q scores
    fig1, ax1 = plt.subplots(figsize=(10, 5))
    ax1.bar(best_assignments_df["Task"], best_assignments_df["Q Score"])
    ax1.set_title("Best Vehicle per Task - VIKOR Q Scores")
    ax1.set_ylabel("Q Score")
    ax1.set_xlabel("Task")
    ax1.tick_params(axis="x", rotation=45)
    fig1.tight_layout()

    chart1_path = "Vehicle_Task_Q_Scores.png"
    fig1.savefig(chart1_path)

    worksheet1 = workbook.add_worksheet("Q_Scores_Chart")
    writer.sheets["Q_Scores_Chart"] = worksheet1
    worksheet1.insert_image("B2", chart1_path)

    # Chart 2: Task weights
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    weights_df.T.plot(kind="bar", ax=ax2)
    ax2.set_title("Task-Specific Criteria Weights")
    ax2.set_ylabel("Weight")
    ax2.set_xlabel("Task")
    ax2.tick_params(axis="x", rotation=45)
    fig2.tight_layout()

    chart2_path = "Task_Weights_Chart.png"
    fig2.savefig(chart2_path)

    worksheet2 = workbook.add_worksheet("Weights_Chart")
    writer.sheets["Weights_Chart"] = worksheet2
    worksheet2.insert_image("B2", chart2_path)

print(f"✅ Output file generated: {output_file}")
print("\nBest Assignments:")
print(best_assignments_df)