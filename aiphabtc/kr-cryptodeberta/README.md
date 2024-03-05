---
license: apache-2.0
---
Korean Pre-Trained Crypto DeBERTa model fine-tuned on BTC sentiment classification dataset.

For more details, check our work [CBITS: Crypto BERT Incorporated Trading System](https://ieeexplore.ieee.org/document/10014986) on IEEE Access.

## Example Use Case: Crypto News BTC Sentiment Classification
```python
from transformers import AutoModelForSequenceClassification, AlbertTokenizer

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

model = AutoModelForSequenceClassification.from_pretrained("LDKSolutions/KR-cryptodeberta-v2-base", num_labels=3) 
model.eval()
model.to(device)

tokenizer = AlbertTokenizer.from_pretrained("LDKSolutions/KR-cryptodeberta-v2-base")

title = "우즈벡, 외국기업의 암호화폐 거래자금 국내계좌 입금 허용" 
content = "비트코인닷컴에 따르면 우즈베키스탄 중앙은행이 외국기업의 국내 은행 계좌 개설 및 암호화폐 거래 자금 입금을 허용했다. 앞서 우즈베키스탄은 외국기업의 은행 계좌 개설 등을 제한 및 금지한 바 있다. 개정안에 따라 이러한 자금은 암호화폐 매입을 위해 거래소로 이체, 혹은 자금이 유입된 관할권 내 등록된 법인 계좌로 이체할 수 있다. 다만 그 외 다른 목적을 위한 사용은 금지된다. 해당 개정안은 지난 2월 9일 발효됐다."


encoded_input = tokenizer(str(title), str(content), max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device) 

with torch.no_grad(): 
    output = model(**encoded_input).logits
    output = nn.Softmax(dim=1)(output) 
    output = output.detach().cpu().numpy()[0] 
    print("호재: {:.2f}% | 악재: {:.2f}% | 중립: {:.2f}%".format(output[0]*100,output[1]*100,output[2]*100)) 
```

## Example Use Case: Crypto News Embedding Similarity
```python
from transformers import AutoModelForSequenceClassification, AlbertTokenizer
from scipy.spatial.distance import cdist 

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") 

model = AutoModel.from_pretrained("LDKSolutions/KR-cryptodeberta-v2-base") 
model.eval()
model.to(device) 

tokenizer = AlbertTokenizer.from_pretrained("LDKSolutions/KR-cryptodeberta-v2-base")

title1 = "USDN 다중담보 자산 전환 제안 통과"
content1 = "웨이브 생태계 스테이블코인 USDN을 다중담보 자산으로 전환하는 제안 투표가 찬성 99%로 오늘 통과됐다. 앞서 코인니스는 웨브가 $WX,$SWOP,$VIRES,$EGG,$WEST를 담보로 해 USDN을 웨이브 생태계 인덱스 자산으로 만들어 USDN 디페깅 이슈를 해결할 플랜을 공개했다고 전한 바 있다."

title2 = "웨이브, USDN 고래 청산안 투표 통과로 30%↑"
content2 = "유투데이에 따르면 웨이브(WAVES) 기반 알고리즘 스테이블코인 뉴트리노(USDN)의 디페그 발생 없이 대규모 USDN 포지션 청산을 가능하게 하는 투표가 만장일치로 통과 됨에 따라 WAVES가 몇시간 안에 30%대 상승폭을 나타냈다. 지난 28일 웨이브 팀이 발표한 USDN의 달러 페그 회복 계획은 다음과 같다.- 커브 및 CRV 토큰으로 USDN 유동성 공급.- 고래 계좌를 청산시켜 Vires 유동성 복구.- USDN 담보물을 두달에 걸쳐 천천히 판매.- 뉴트리노 프로토콜 자본 조달을 위한 새로운 토큰 발행."

encoded_input1 = tokenizer(str(title1), str(content1), max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)
encoded_input2 = tokenizer(str(title2), str(content2), max_length=512, padding="max_length", truncation=True, return_tensors="pt").to(device)

with torch.no_grad(): 
    emb1 = model(**encoded_input1)[0][:,0,:].detach().cpu().numpy() 
    emb2 = model(**encoded_input2)[0][:,0,:].detach().cpu().numpy() 
    sim_scores = cdist(emb1, emb2, "cosine")[0] 
print(f"cosine distance = {sim_scores[0]}")
```

