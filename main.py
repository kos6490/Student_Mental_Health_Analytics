# Data source : kaggle - Student Mental health
# Data URL : https://www.kaggle.com/datasets/shariful07/student-mental-health

# File load check

import pandas as pd

data_file = pd.read_csv('student_mental_health.csv')

print(data_file.head())
print(data_file.info()) 