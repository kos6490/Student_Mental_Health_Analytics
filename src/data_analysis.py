import pandas as pd


# 데이터 분석
# 학년과 우울증/불안의 관계
def analyze_depression_anxiety_by_year(df):
    df_copy = df.copy()

    df_copy["Depression_Numeric"] = df_copy["Depression"].apply(
        lambda x: 1 if x == "Yes" else 0
    )  # Yes or No를 숫자로 변환

    df_copy["Anxiety_Numeric"] = df_copy["Anxiety"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    result = (
        df_copy.groupby("Year of Study_Numeric")[
            ["Depression_Numeric", "Anxiety_Numeric"]
        ].mean()
        * 100
    )

    return result.sort_index()


# 성적과 공황발작의 관계
def analyze_panic_by_cgpa(df):
    df_copy = df.copy()

    df_copy["Panic Attack_Numeric"] = df_copy["Panic Attack"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    result = df_copy.groupby("CGPA")[["Panic Attack_Numeric"]].mean() * 100

    return result.sort_index()


# 성적과 우울증의 관계
def analyze_depression_by_cgpa(df):
    result = df.groupby("Depression")["CGPA_Numeric"].describe()

    return result


# 전공과 불안의 관계
def analyze_anxiety_by_course(df):
    df_copy = df.copy()

    df_copy["Anxiety_Numeric"] = df_copy["Anxiety"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    top5_courses = (
        df_copy["Course"].value_counts().head().index
    )  # 가장 많은 전공 상위 5개를 대상
    df_course = df_copy[df_copy["Course"].isin(top5_courses)]

    result = df_course.groupby("Course")[["Anxiety_Numeric"]].mean() * 100

    return result.sort_values(by="Anxiety_Numeric", ascending=False)


# 공황발작과 치료 받은 비율의 관계
def analyze_treatment_by_panic(df):
    df_copy = df.copy()

    df_copy["Treatment_Numeric"] = df_copy["Treatment"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    result = df_copy.groupby("Panic Attack")["Treatment_Numeric"].agg(["mean", "count"])
    result["mean"] = result["mean"] * 100  # 백분율 반환
    result.columns = ["Treatment Rate(%)", "Student Count"]

    return result
