import pandas as pd
import numpy as np
import os

# 1. 데이터 불러오기
# 경로는 사용자의 실제 파일 경로에 맞게 수정
file_path = '/Users/namchaewon/Desktop/python/AI_contest/debt_collection_dummy_data.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 경로를 다시 확인해주세요.")
    exit()

# 데이터프레임의 첫 5행 출력
print("\n[원본 데이터프레임]")
print(df.head())

# 2. '1개월 미만' 채무자 제외 (연체일수 30일 미만)
# 연체일수(days_past_due)가 30일 이상인 데이터만 필터링
# 1개월 = 30일, 2개월 = 60일, 3개월 = 90일 등으로 간주하여 분류
df_filtered = df[df['days_past_due'] >= 30].copy()
print(f"\n[필터링 결과] 1개월 미만 채무자 제외 후 데이터 개수: {len(df_filtered)}")

# 3. 변수 정제 및 결측치 처리 (기존과 동일)
df_filtered['amount_recovered'] = df_filtered['amount_recovered'].fillna(0)
df_filtered['outstanding_amount'] = df_filtered['outstanding_amount'].fillna(df_filtered['original_amount'])
df_filtered['recovery_time_days'] = df_filtered.apply(
    lambda row: row['recovery_time_days'] if row['collection_status'] == '성공' else np.nan,
    axis=1
)

# 4. 핵심 변수 생성 (Feature Engineering)

# 4-1. '연체기간_그룹' 변수 생성 (요청 기준에 따라 수정)
# 1개월(30일), 2개월(60일), 3개월(90일), 6개월(180일)을 기준으로 분류
bins = [30, 90, 180, 366]
labels = ['단기', '중기', '상각']
df_filtered['days_past_due_group'] = pd.cut(df_filtered['days_past_due'], bins=bins, labels=labels, right=False)

# 4-2. '회수율' 변수 생성
df_filtered['collection_rate'] = np.where(df_filtered['original_amount'] > 0, df_filtered['amount_recovered'] / df_filtered['original_amount'], 0)

# 4-3. '접촉효율성' 변수 생성
df_filtered['contact_efficiency'] = np.where(df_filtered['contact_attempts'] > 0, df_filtered['amount_recovered'] / df_filtered['contact_attempts'], 0)

# 5. 최종 데이터 확인
print("\n[전처리 후 데이터프레임 (필터링 및 그룹화 적용)]")
print(df_filtered[['days_past_due', 'days_past_due_group', 'collection_rate', 'contact_efficiency']].head())
print("\n[그룹별 데이터 개수]")
print(df_filtered['days_past_due_group'].value_counts())

# 최종 전처리된 데이터를 새로운 CSV 파일로 저장
preprocessed_file_path = '/Users/namchaewon/Desktop/python/AI_contest/preprocessed_debt_data_filtered_revised.csv'
os.makedirs(os.path.dirname(preprocessed_file_path), exist_ok=True)
df_filtered.to_csv(preprocessed_file_path, index=False, encoding='utf-8-sig')
print(f"\n필터링 및 전처리된 데이터가 성공적으로 저장되었습니다: {preprocessed_file_path}")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import dendrogram, linkage
import matplotlib.pyplot as plt
import os

# 1. 전처리된 데이터 불러오기
file_path = '/Users/namchaewon/Desktop/python/AI_contest/preprocessed_debt_data_filtered_revised.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("전처리된 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 전 단계에서 생성한 파일명을 확인해주세요.")
    exit()

# 2. 군집 분석에 사용할 변수 선택 및 스케일링
features = ['age', 'original_amount', 'days_past_due', 'contact_attempts', 
            'credit_score', 'income_level', 'collection_rate']
df_clustering = df[features].copy()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)
print("\n[군집 분석을 위한 데이터 표준화 완료]")

# 3. 계층적 군집 분석 및 덴드로그램 생성
# 'ward' 방식은 군집 내 분산이 가장 적게 증가하는 방향으로 군집을 병합
linked = linkage(df_scaled, method='ward')

plt.figure(figsize=(15, 8))
dendrogram(linked, orientation='top', distance_sort='descending', show_leaf_counts=True)
plt.title('Hierarchical Clustering Dendrogram')
plt.xlabel('Sample Index')
plt.ylabel('Distance')
plt.show()

print("\n위 덴드로그램을 참고하여 비즈니스 목적에 맞는 군집 수를 결정하세요.")
print("덴드로그램에서 y축(거리)의 특정 지점을 수평선으로 자른다고 생각했을 때,")
print("수평선 아래에 생성되는 세로선 그룹의 개수가 군집의 수가 됩니다.")

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# 1. 전처리된 데이터 불러오기
file_path = '/Users/namchaewon/Desktop/python/AI_contest/preprocessed_debt_data_filtered_revised.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("전처리된 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 전 단계에서 생성한 파일명을 확인해주세요.")
    exit()

# 2. 군집 분석에 사용할 변수 선택 및 스케일링
features = ['age', 'original_amount', 'days_past_due', 'contact_attempts', 
            'credit_score', 'income_level', 'collection_rate']
df_clustering = df[features].copy()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df_clustering)
df_scaled = pd.DataFrame(df_scaled, columns=features)
print("\n[군집 분석을 위한 데이터 표준화 완료]")

# 3. K-Means 모델 학습 (덴드로그램 기반 최적의 K=3으로 설정)
optimal_k = 3
print(f"\n덴드로그램 분석을 통해 결정된 최적의 K={optimal_k}로 군집 분석을 진행합니다.")

kmeans = KMeans(n_clusters=optimal_k, random_state=42, n_init='auto')
df['cluster'] = kmeans.fit_predict(df_scaled)

# 4. 군집 특성 분석
# 각 군집별 평균 특성 확인 (비율 변수는 소수점 둘째 자리까지 표시)
cluster_summary = df.groupby('cluster')[features].mean()

# 스케일링 전 원본 데이터의 평균으로 복원하여 가독성 향상
cluster_summary = pd.DataFrame(scaler.inverse_transform(cluster_summary), columns=features)
cluster_summary['cluster'] = cluster_summary.index
cluster_summary = cluster_summary.round(2)
print("\n[군집별 주요 특성 요약]")
print(cluster_summary)

# 각 군집에 속한 데이터 개수 확인
cluster_counts = df['cluster'].value_counts().sort_index()
print("\n[군집별 데이터 개수]")
print(cluster_counts)

# 군집 특성에 따른 군집 이름 부여
# 각 클러스터의 평균값을 보고 의미있는 이름으로 재분류합니다.
# 이 부분은 데이터에 따라 달라지므로, 출력된 결과를 보고 직접 정의할 수 있습니다.
# 예를 들어:
# cluster_names = {0: '단기 연체 저위험군', 1: '장기 연체 고액 채무자', 2: '중기 연체 중위험군'}
# df['cluster_name'] = df['cluster'].map(cluster_names)

# 5. 군집 결과를 CSV 파일로 저장
clustered_file_path = '/Users/namchaewon/Desktop/python/AI_contest/clustered_debt_data.csv'
df.to_csv(clustered_file_path, index=False, encoding='utf-8-sig')
print(f"\n군집 분석 결과가 포함된 데이터가 '{clustered_file_path}' 파일로 저장되었습니다.")

import pandas as pd
import os

# 1. 군집 분석이 완료된 데이터 불러오기
file_path = '/Users/namchaewon/Desktop/python/AI_contest/clustered_debt_data.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("군집 분석이 완료된 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 전 단계에서 생성한 파일명을 확인해주세요.")
    exit()

# 2. 군집별 주요 특성 분석 및 이름 정의
# 각 군집의 평균값을 계산하여 특성 파악
cluster_summary = df.groupby('cluster').agg(
    avg_original_amount=('original_amount', 'mean'),
    avg_days_past_due=('days_past_due', 'mean'),
    avg_collection_rate=('collection_rate', 'mean'),
    avg_contact_attempts=('contact_attempts', 'mean'),
    count=('debtor_id', 'count')
).round(2)

print("\n[군집별 주요 특성 분석 결과]")
print(cluster_summary)

import pandas as pd
import os

# 1. 군집 분석이 완료된 데이터 불러오기
file_path = '/Users/namchaewon/Desktop/python/AI_contest/clustered_debt_data.csv'

try:
    df = pd.read_csv(file_path, encoding='utf-8-sig')
    print("군집 분석이 완료된 데이터를 성공적으로 불러왔습니다.")
except FileNotFoundError:
    print("오류: 파일을 찾을 수 없습니다. 전 단계에서 생성한 파일명을 확인해주세요.")
    exit()

# 2. 군집별 특성 분석 및 이름, 전략 정의 (수치 기반으로 재정의)
# 클러스터별 평균 연체일수, 원금, 회수율 등을 기반으로 특성 정의
# 클러스터 0: avg_days_past_due가 234일(중간), avg_original_amount가 31M(가장 높음), avg_collection_rate이 0.2(가장 낮음) -> 고액 채무, 낮은 회수율
# 클러스터 1: avg_days_past_due가 277일(가장 김), avg_original_amount가 22M(중간), avg_contact_attempts가 9.92(가장 많음) -> 장기 연체, 높은 관리 노력 필요
# 클러스터 2: avg_days_past_due가 106일(가장 짧음), avg_collection_rate가 0.57(가장 높음) -> 단기 연체, 높은 회수 가능성

# 위 분석을 기반으로 군집별 이름과 전략 매핑
cluster_names = {
    0: '고액_고위험군',
    1: '중위험_장기연체군',
    2: '저위험_단기연체군'
}

strategy_map = {
    '고액_고위험군': '전담 추심원 배정 및 법적 절차 준비',
    '중위험_장기연체군': '채무 조정(분할 상환, 유예) 제안',
    '저위험_단기연체군': '자동 문자/이메일 발송'
}

# 3. 데이터프레임에 군집 이름 및 전략 추가
df['cluster_name'] = df['cluster'].map(cluster_names)
df['recommended_strategy'] = df['cluster_name'].map(strategy_map)

# 4. 최종 결과 확인
print("\n[전략 매핑 완료된 데이터프레임]")
print(df[['debtor_id', 'cluster', 'cluster_name', 'recommended_strategy', 'days_past_due', 'original_amount', 'collection_rate']].head())

# 5. 최종 데이터셋 저장
final_file_path = '/Users/namchaewon/Desktop/python/AI_contest/final_debt_data.csv'
df.to_csv(final_file_path, index=False, encoding='utf-8-sig')
print(f"\n군집 정보 및 전략이 포함된 최종 데이터가 '{final_file_path}' 파일로 저장되었습니다.")
