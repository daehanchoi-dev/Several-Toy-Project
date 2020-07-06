"""
import pandas as pd
import pandas_profiling
df = pd.read_csv('dataset_spam.csv',encoding='latin1')
#  df를 프로파일링한 리포트 생성
pr =df.profile_report()
pr.to_file('./pr_report.html')
"""



from konlpy.tag import Okt
okt = Okt()
print(okt.morphs("열심히 일한 당신, 연휴에는 쉬세요"))




