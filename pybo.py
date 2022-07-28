from flask import Flask, make_response, jsonify, request

# torch
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import gluonnlp as nlp
import numpy as np
from tqdm import tqdm, tqdm_notebook

# kobert
from kobert.utils import get_tokenizer
from kobert.pytorch_kobert import get_pytorch_kobert_model

import requests
import json

app = Flask(__name__)

# GPU, CPU 사용
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# BERT 모델, Vocabulary 불러오기 필수
bertmodel, vocab = get_pytorch_kobert_model()


# KoBERT에 입력될 데이터셋 정리
class BERTDataset(Dataset):
    def __init__(self, dataset, sent_idx, label_idx, bert_tokenizer, max_len,
                 pad, pair):
        transform = nlp.data.BERTSentenceTransform(
            bert_tokenizer, max_seq_length=max_len, pad=pad, pair=pair)

        self.sentences = [transform([i[sent_idx]]) for i in dataset]
        self.labels = [np.int32(i[label_idx]) for i in dataset]

    def __getitem__(self, i):
        return (self.sentences[i] + (self.labels[i],))

    def __len__(self):
        return (len(self.labels))

    # 모델 정의


class BERTClassifier(nn.Module):  ## 클래스를 상속
    def __init__(self,
                 bert,
                 hidden_size=768,
                 num_classes=4,  ##클래스 수 조정##
                 dr_rate=None,
                 params=None):
        super(BERTClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:v] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)

        _, pooler = self.bert(input_ids=token_ids, token_type_ids=segment_ids.long(),
                              attention_mask=attention_mask.float().to(token_ids.device))
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        return self.classifier(out)


# Setting parameters
max_len = 100
batch_size = 32
warmup_ratio = 0.1
num_epochs = 10
max_grad_norm = 1
log_interval = 100
learning_rate = 5e-5

## 학습 모델 로드
PATH = 'bert'
# model = torch.load(PATH + 'KoBERT_대화.pt')  # 전체 모델을 통째로 불러옴, 클래스 선언 필수
model = BERTClassifier(bertmodel)
model.load_state_dict(torch.load(PATH + 'model_state_dict.pt', map_location='cpu'))  # state_dict를 불러 온 후, 모델에 저장
model.to(device)

# 토큰화
tokenizer = get_tokenizer()
tok = nlp.data.BERTSPTokenizer(tokenizer, vocab, lower=False)


def new_softmax(a):
    c = np.max(a)  # 최댓값
    exp_a = np.exp(a - c)  # 각각의 원소에 최댓값을 뺀 값에 exp를 취한다. (이를 통해 overflow 방지)
    sum_exp_a = np.sum(exp_a)
    y = (exp_a / sum_exp_a) * 100
    return np.round(y, 3)


# In[ ]:
def predict(predict_sentence):
    data = [predict_sentence, '0']
    dataset_another = [data]

    another_test = BERTDataset(dataset_another, 0, 1, tok, max_len, True, False)
    test_dataloader = torch.utils.data.DataLoader(another_test, batch_size=batch_size, num_workers=5)

    model.eval()

    for batch_id, (token_ids, valid_length, segment_ids, label) in enumerate(test_dataloader):
        token_ids = token_ids.long().to(device)
        segment_ids = segment_ids.long().to(device)

        valid_length = valid_length
        label = label.long().to(device)

        out = model(token_ids, valid_length, segment_ids)

        test_eval = []
        for i in out:
            logits = i
            logits = logits.detach().cpu().numpy()
            min_v = min(logits)
            total = 0
            probability = []
            logits = np.round(new_softmax(logits), 3).tolist()
            for logit in logits:
                print(logit)
                probability.append(np.round(logit, 3))

            if np.argmax(logits) == 0:
                emotion = '공포'
            elif np.argmax(logits) == 1:
                emotion = '분노'
            elif np.argmax(logits) == 2:
                emotion = '슬픔'
            elif np.argmax(logits) == 3:
                emotion = '행복'

            probability.append(emotion)
            print(probability)
    return emotion


@app.route('/')
def hello_world():
    return 'Hello World!'


# 예측 모델 설정
@app.route('/emotion', methods=['POST'])
def hello():
    content = request.json.get('content')
    print(content)
    res = predict(content)
    # res = f"0:{res[0]}<br>1:{res[1]}<br>2:{res[2]}<br>3:{res[3]}"
    return res


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=80, debug=True)
