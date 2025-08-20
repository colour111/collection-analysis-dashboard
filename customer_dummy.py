import pandas as pd
import numpy as np
from faker import Faker
import random

# Faker 라이브러리를 한국어로 설정
fake = Faker('ko_KR')

# 더미데이터 생성
num_data = 300
data = {
    'debtor_id': range(1, num_data + 1),
    'age': np.random.randint(20, 70, size=num_data),
    'gender': np.random.choice(['남성', '여성'], size=num_data),
    'region': [fake.city() for _ in range(num_data)],
    'original_amount': np.random.randint(100000, 50000000, size=num_data),
    'days_past_due': np.random.randint(1, 365, size=num_data)
}

df = pd.DataFrame(data)

# 미회수액, 회수금액, 회수 성공 여부 등 종속적인 데이터 생성
def calculate_collection_status(row):
    # 연체 기간이 길수록 회수율이 낮아지는 경향을 반영
    success_rate = 1.0 - (row['days_past_due'] / 365) * 0.7  # 연체 기간에 따라 성공률 조정
    return '성공' if random.random() < success_rate else '실패'

df['collection_status'] = df.apply(calculate_collection_status, axis=1)
df['amount_recovered'] = df.apply(
    lambda row: np.random.randint(row['original_amount'] * 0.5, row['original_amount']) if row['collection_status'] == '성공' else 0,
    axis=1
)
df['outstanding_amount'] = df['original_amount'] - df['amount_recovered']

# 추심 활동 정보 및 예측 모델 변수 생성
df['strategy_used'] = df.apply(
    lambda row: np.random.choice(['전화', '방문']) if row['days_past_due'] > 90 else np.random.choice(['문자', '전화']),
    axis=1
)
df['contact_attempts'] = df['days_past_due'].apply(
    lambda x: int(np.random.normal(x / 30, 1)) + 1
)
df['contact_success_count'] = df['contact_attempts'].apply(
    lambda x: int(np.random.beta(a=0.5, b=5) * x)
)
df['recovery_time_days'] = df.apply(
    lambda row: np.random.randint(1, row['days_past_due']) if row['collection_status'] == '성공' else np.nan,
    axis=1
)

# 예측 모델을 위한 추가 정보
df['credit_score'] = np.random.randint(400, 950, size=num_data)
df['income_level'] = np.random.randint(2000, 8000, size=num_data)
df['employment_status'] = np.random.choice(['직장인', '자영업', '무직'], size=num_data, p=[0.7, 0.2, 0.1])

# 불필요한 값 조정 및 데이터 형식 변경
df['contact_attempts'] = df['contact_attempts'].clip(lower=1)
df['contact_success_count'] = df['contact_success_count'].clip(lower=0)

# 최종 데이터 확인
print(df.head())
print(df.info())

# CSV 파일로 저장
df.to_csv('debt_collection_dummy_data.csv', index=False)