# Minimal BERT Implementation

## Caravan Research PaperStudy Team - Alki

This is a minimal BERT implementation built from scratch using only NumPy with GPT4o.

이 프로젝트는 GPT4o를 이용하여 NumPy만 사용하여 BERT를 처음부터 구현한 최소 구조의 버전입니다. CPU만 사용할 수 있습니다.

 
---

## Structure / 구성

- `computation/` : Tensor class and autograd engine  
- `model/`       : BERT layers, training loop, utilities  
- `tokenization/`: WordPiece tokenizer, input formatting  
- `dataset/`     : Dataset loader (e.g., WikiText-2)  
- `train.py`     : Training entrypoint  
- `trainer.py`   : Separated training loop  

---

## Features / 기능

- Pure NumPy implementation (no external frameworks)  
- Custom autograd engine  
- WordPiece tokenizer  
- Multi-head attention, feedforward layers, layer norm  
- MLM / NSP loss  
- Model saving / inference support  

---

## Run / 실행

```bash
python train.py



## License

This project is licensed under the MIT License.