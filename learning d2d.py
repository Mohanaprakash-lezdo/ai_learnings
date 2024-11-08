with open(r'C:/Users/mohanaprakash.n/Documents/the-verdict.txt',encoding='utf-8')as f:
    raw_text=f.read()
print(len(raw_text))
import re
result=re.split(r'([,.;:?!_"()\'] |--|\s)', raw_text)
result=[item.strip() for item in result if item.strip()]
print(len(result))
all_words=sorted(set(result))
vocal_size=len(all_words)
print(vocal_size)
vocab={token:integer for integer,token in enumerate(all_words)}
print (vocab)