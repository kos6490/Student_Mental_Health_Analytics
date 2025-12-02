# Data source : kaggle - Student Mental health
# Data URL : https://www.kaggle.com/datasets/shariful07/student-mental-health

# File load check

import pandas as pd
from src import data_processing
from src import data_analysis
from src import data_visualization
from src import machine_learning

# 데이터 로딩
print("\n[Step 1] Loading Data...")
data_file = pd.read_csv("data/student_mental_health.csv")

# 데이터 전처리
print("\n[Step 2] Preprocessing Data...")
data_file = data_processing.processing(data_file)

# 데이터 분석
print("\n[Step 3] Analyzing Data...")
print("\n[학년과 우울증/불안의 관계]")
print(data_analysis.analyze_depression_anxiety_by_year(data_file))
print()

print("[성적과 공황발작의 관계]")
print(data_analysis.analyze_panic_by_cgpa(data_file))
print()

print("[성적과 우울증의 관계]")
print(data_analysis.analyze_depression_by_cgpa(data_file))
print()

print("[전공과 불안의 관계]")
print(data_analysis.analyze_anxiety_by_course(data_file))
print()

print("[공황발작과 치료 받은 비율의 관계]")
print(data_analysis.analyze_treatment_by_panic(data_file))
print()

# 데이터 시각화
print("\n[Step 4] Visualizing Data...")
print("\n[분석 결과 그래프]")
data_visualization.plot_graphs(data_file)

# 머신러닝
print("\n[Step 5] Machine Learning (Risk Prediction)...")
accuracy, report, importance = machine_learning.predict_risk(data_file)

print(f"\n🎯 예측 정확도 (Accuracy) : {accuracy*100:.2f}%")
print("\n머신러닝 결과 (Report)")
print(report)

top_3 = importance.sort_values(ascending=False).head(3)
print("\n🚨 Student Mental Health Risk의 주요 요인 (Top 3)")
print(top_3)
