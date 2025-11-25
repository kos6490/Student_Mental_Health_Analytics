import pandas as pd
import numpy as np

# 학년이 Y/year n 형식이므로 n을 추출해 사용
def clean_year(year_str):
    return int(str(year_str).lower().replace('year', '').strip())

# CGPA가 최솟값 - 최댓값 형식이므로 평균을 구해 사용
def clean_cgpa(cgpa_range):
    temp_cgpa = str(cgpa_range).strip().split('-')

    if(len(temp_cgpa)) == 2:
        return (float(temp_cgpa[1]) - (temp_cgpa[0])) / 2
    else:
        return float(temp_cgpa[0])

# 데이터프레임을 받아 데이터를 전처리
def processing(df):
    df_copy = df.copy() # 원본 보호
    new_columns = ['Timestamp', 'Gender', 'Age', 'Course', 'Year of Study', 'CGPA', 'Marital Status', 'Depression', 'Anxiety', 'Panic Attack', 'Treatment'] # 간단한 열 이름

    df_copy.columns = new_columns # 간단한 열 이름으로 변경
    df_copy.dropna(inplace=True) # 결측치 제거

    df_copy['Year of Study_Numeric'] = df_copy['Year of Study'].apply(clean_year)
    df_copy['CGPA_Numeric'] = df_copy['CGPA'].apply(clean_cgpa)

    return df_copy