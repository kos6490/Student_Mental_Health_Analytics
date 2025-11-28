import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import platform

# - 기호 깨짐 방지
plt.rc("axes", unicode_minus=False)

# 그래프 이미지를 저장할 폴더가 없다면 생성
if not os.path.exists("output"):
    os.makedirs("output")


# 그래프 이미지를 저장
def save_plot(fig, filename):
    fig.savefig(f"output/{filename}", bbox_inches="tight", dpi=300)
    print(f"✅ [저장 완료] output/{filename}")

    plt.close(fig)


# 5가지 데이터 분석 결과를 그래프로 그리기
def plot_graphs(df):
    df_copy = df.copy()

    # 그래프를 그리기 위해 Yes/No를 1/0으로 변경
    df_copy["Depression_Numeric"] = df_copy["Depression"].apply(
        lambda x: 1 if x == "Yes" else 0
    )
    df_copy["Anxiety_Numeric"] = df_copy["Anxiety"].apply(
        lambda x: 1 if x == "Yes" else 0
    )
    df_copy["Panic Attack_Numeric"] = df_copy["Panic Attack"].apply(
        lambda x: 1 if x == "Yes" else 0
    )

    # 그래프 배경 설정
    sns.set_style("whitegrid")

    # 한글 폰트 깨짐 방지
    if platform.system() == "Windows":
        plt.rc("font", family="Malgun Gothic")  # Windows
    elif platform.system() == "Darwin":
        plt.rc("font", family="AppleGothic")  # Mac
    else:
        plt.rc("font", family="NanumGothic")  # 그 외 OS는 별도

    # 그래프 1. 학년과 우울증/불안의 관계
    fig, ax = plt.subplots(figsize=(10, 6))

    year_avg = (
        df_copy.groupby("Year of Study_Numeric")[
            ["Depression_Numeric", "Anxiety_Numeric"]
        ].mean()
        * 100
    )

    sns.lineplot(data=year_avg, markers=True, linewidth=3, ax=ax)
    ax.set_title("학년별 우울증/불안 비율", fontsize=16)
    ax.set_ylabel("비율 (%)", fontsize=12)
    ax.set_xlabel("학년 (Year)", fontsize=12)
    ax.set_xticks([1, 2, 3, 4])

    save_plot(fig, "Graph 1 - Relationship between year and depression-anxiety.png")

    # 그래프 2. 성적과 공황발작의 관계
    fig, ax = plt.subplots(figsize=(10, 6))

    cgpa_avg = df_copy.groupby("CGPA")["Panic Attack_Numeric"].mean().sort_index() * 100

    sns.barplot(
        x=cgpa_avg.index,
        y=cgpa_avg.values,
        hue=cgpa_avg.index,
        legend=False,
        palette="Reds",
        ax=ax,
    )
    ax.set_title("성적별 공황발작 비율", fontsize=16)
    ax.set_ylabel("비율 (%)", fontsize=12)
    ax.set_xlabel("성적 (CGPA)", fontsize=12)

    save_plot(fig, "Graph 2 - Relationship between CGPA and panic attack.png")

    # 그래프 3. 성적과 우울증의 관계
    fig, ax = plt.subplots(figsize=(8, 6))

    sns.boxplot(
        x="Depression",
        y="CGPA_Numeric",
        data=df_copy,
        hue="Depression",
        legend=False,
        palette="Set2",
        ax=ax,
    )
    ax.set_title("우울증 유무에 따른 CGPA", fontsize=16)
    ax.set_ylabel("성적 (CGPA)", fontsize=12)

    save_plot(fig, "Graph 3 - Relationship between CGPA and depression.png")

    # 그래프 4. 전공과 불안의 관계
    fig, ax = plt.subplots(figsize=(12, 6))

    top5_courses = df_copy["Course"].value_counts().head().index
    df_course = df_copy[df_copy["Course"].isin(top5_courses)]

    course_avg = (
        df_course.groupby("Course")["Anxiety_Numeric"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    sns.barplot(
        x=course_avg.values,
        y=course_avg.index,
        hue=course_avg.index,
        legend=False,
        palette="viridis",
        ax=ax,
    )
    ax.set_title("전공별 불안 비율 (Top 5)", fontsize=16)
    ax.set_xlabel("비율 (%)", fontsize=12)

    save_plot(fig, "Graph 4 - Relationship between course and anxiety.png")

    # 그래프 5. 공황발작과 치료 받은 비율의 관계
    fig, ax = plt.subplots(figsize=(8, 8))

    panic_avg = df[df["Panic Attack"] == "Yes"]
    treatment_avg = panic_avg["Treatment"].value_counts()

    ax.pie(
        treatment_avg,
        labels=treatment_avg.index,
        autopct="%1.1f%%",
        startangle=90,
        colors=["#ff9999", "#66b3ff"],
    )
    ax.set_title("공황발작 경험 학생의 치료 여부", fontsize=16)

    save_plot(fig, "Graph 5 - Relationship between panic attack and treatment rate.png")

    print("\n✅ 모든 그래프 생성이 완료되었습니다! (output 폴더 확인)")
