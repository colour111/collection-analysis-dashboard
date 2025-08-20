import streamlit as st
import pandas as pd
import plotly.express as px
import os

# 1. 페이지 설정
st.set_page_config(
    page_title="채권추심 데이터 대시보드",
    layout="wide",
    initial_sidebar_state="expanded",
)

# 2. 데이터 불러오기
@st.cache_data
def load_data(file_path):
    """
    데이터를 불러오는 함수. 캐시를 사용하여 데이터를 한 번만 로드합니다.
    Args:
        file_path (str): CSV 파일 경로
    Returns:
        pd.DataFrame: 불러온 데이터프레임
    """
    try:
        df = pd.read_csv(file_path, encoding='utf-8-sig')
        return df
    except FileNotFoundError:
        st.error(f"오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
        return None

# 데이터 파일 경로 설정
data_file_path = '/Users/namchaewon/Desktop/python/AI_contest/final_debt_data.csv'
df = load_data(data_file_path)

if df is not None:
    st.title('💰 채권추심 데이터 대시보드')

    # 3. 메인 화면 상단에 버튼 형식으로 군집 선택 기능 구현
    cluster_names = df['cluster_name'].unique().tolist()
    
    # '전체' 버튼 추가
    all_option = '전체'
    buttons = [all_option] + cluster_names
    
    selected_cluster = st.radio(
        '군집 선택:',
        buttons,
        horizontal=True
    )
    
    # 선택된 군집에 따라 데이터 필터링
    if selected_cluster == all_option:
        df_filtered = df.copy()
    else:
        df_filtered = df[df['cluster_name'] == selected_cluster]
    
    if df_filtered.empty:
        st.warning('선택된 군집에 해당하는 데이터가 없습니다.')

    else:
        # 4. KPI 지표 표시
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_debt = df_filtered['original_amount'].sum()
            st.metric(label="총 채권 원금", value=f"{total_debt:,.0f} 원")

        with col2:
            total_recovered = df_filtered['amount_recovered'].sum()
            st.metric(label="총 회수 금액", value=f"{total_recovered:,.0f} 원")
            
        with col3:
            # 총 채권 원금이 0인 경우를 방지
            if df_filtered['original_amount'].sum() > 0:
                total_rate = (df_filtered['amount_recovered'].sum() / df_filtered['original_amount'].sum()) * 100
            else:
                total_rate = 0
            st.metric(label="총 회수율", value=f"{total_rate:.2f}%")

        st.markdown('---')

        # 5. 시각화 섹션
        st.header('군집별 특성 분석')
        
        # 5-1. 군집별 데이터 개수 및 회수율
        fig_pie = px.pie(
            df_filtered,
            names='cluster_name',
            title='군집별 채무자 비율',
            color='cluster_name',
            hole=0.4,
            width=500,
            height=500
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        
        st.plotly_chart(fig_pie, use_container_width=True)

        # 5-2. 군집별 평균 특성 바 차트
        st.subheader('군집별 평균 특성 비교')
        
        # 필터링된 데이터의 그룹 요약
        cluster_summary = df_filtered.groupby('cluster_name').agg(
            avg_original_amount=('original_amount', 'mean'),
            avg_days_past_due=('days_past_due', 'mean'),
            avg_collection_rate=('collection_rate', 'mean')
        ).reset_index()

        fig_bar = px.bar(
            cluster_summary,
            x='cluster_name',
            y=['avg_original_amount', 'avg_days_past_due', 'avg_collection_rate'],
            barmode='group',
            title='군집별 평균 채권원금, 연체일수, 회수율',
            labels={
                'value': '평균값',
                'variable': '특성',
                'cluster_name': '군집'
            },
            color_discrete_map={
                'avg_original_amount': '#4B0082',
                'avg_days_past_due': '#8A2BE2',
                'avg_collection_rate': '#DA70D6'
            }
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        st.markdown('---')

        # 6. 추천 전략 및 상세 정보 테이블
        st.header('군집별 추천 회수 전략')
        
        strategy_df = df_filtered.groupby('cluster_name')['recommended_strategy'].first().reset_index()
        strategy_df.columns = ['군집', '추천 전략']
        st.table(strategy_df)

        st.markdown('---')
        st.header('개별 채무자 정보')
        st.dataframe(df_filtered)
