# Data source : kaggle - Student Mental health
# Data URL : https://www.kaggle.com/datasets/shariful07/student-mental-health

# File load check

import pandas as pd
from src import data_processing
from src import data_analysis

# 데이터 로딩
data_file = pd.read_csv("data/student_mental_health.csv")

# 데이터 전처리
data_file = data_processing.processing(data_file)

# 데이터 분석
print("[학년과 우울증/불안의 관계]")
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
