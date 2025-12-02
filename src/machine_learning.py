import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# 머신러닝 결과 이미지를 저장할 폴더가 없다면 생성
if not os.path.exists("output"):
    os.makedirs("output")


# Depression | Anxiety | Panic 을 고려하여 Mental Health Risk를 예측
def predict_risk(df):
    model_df = df.copy()

    model_df["Risk"] = model_df.apply(
        lambda x: (
            1
            if (x["Depression"] == "Yes")
            or (x["Anxiety"] == "Yes")
            or (x["Panic Attack"] == "Yes")
            else 0
        ),
        axis=1,
    )

    le = LabelEncoder()
    columns = ["Gender", "Course", "Marital Status", "Treatment"]

    for column in columns:
        model_df[column] = le.fit_transform(model_df[column])

    y = model_df["Risk"]

    drop_columns = [
        "Risk",
        "Depression",
        "Anxiety",
        "Panic Attack",
        "Timestamp",
        "CGPA",
        "Year of Study",
        "Treatment",
        "Gender",
        "Marital Status",
    ]

    x = model_df.drop(drop_columns, axis=1)

    print(f"\n학습에 사용되는 특성 : {list(x.columns)}")

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.3, random_state=42
    )

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(x_train, y_train)

    y_predict = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_predict)
    report = classification_report(y_test, y_predict)

    matrix = confusion_matrix(y_test, y_predict)
    plt.figure(figsize=(6, 5))
    sns.heatmap(matrix, annot=True, fmt="d", cmap="Reds", cbar=False)
    plt.xlabel("Predicted Risk (by AI)")
    plt.ylabel("Actual Risk")
    plt.title("Mental Health Risk")
    plt.savefig("output/Report 1 - Mental Health Risk Prediction.png")
    print("\n✅ 머신러닝 결과 이미지 생성이 완료되었습니다! (output 폴더 확인)")
    plt.close()

    importances = model.feature_importances_
    feature_names = x.columns
    feature_imp = pd.Series(
        importances, index=feature_names.sort_values(ascending=False)
    )

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_imp.values, y=feature_imp.index, palette="viridis")
    plt.title("Student Mental Health Risk의 주요 요인")
    plt.xlabel("Importance")
    plt.savefig("output/Report 2 - Importances.png")
    print("\n✅ 머신러닝 결과 이미지 생성이 완료되었습니다! (output 폴더 확인)")
    plt.close()

    return accuracy, report, feature_imp
