"""
import pandas as pd
import pandas_profiling
df = pd.read_csv('dataset_spam.csv',encoding='latin1')
#  df를 프로파일링한 리포트 생성
pr =df.profile_report()
pr.to_file('./pr_report.html')
"""

"""

import nltk
from nltk.tokenize import word_tokenize
text = "I am actively looking for Ph.D. Students. and you are a Ph.D. student."
print(word_tokenize(text))


from nltk.tag import pos_tag
x = word_tokenize(text)
pos_tag(x)
print(pos_tag(x))

"""
import konlpy
from konlpy.tag import Okt
okt = Okt()
print(okt.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요!")) # 형태소 추출
print(okt.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요!")) # 품사 태깅
print(okt.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요!")) # 명사 추출

from konlpy.tag import Kkma
kkma = Kkma()
print(kkma.morphs("열심히 코딩한 당신, 연휴에는 여행을 가봐요!"))
print(kkma.pos("열심히 코딩한 당신, 연휴에는 여행을 가봐요!"))
print(kkma.nouns("열심히 코딩한 당신, 연휴에는 여행을 가봐요!"))



