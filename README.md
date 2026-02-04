# Docker 환경 기반 Parameter Server 분산 딥러닝

본 프로젝트는 Docker 컨테이너 환경에서 PyTorch RPC를 활용하여 Parameter Server 기반 비동기 분산 딥러닝을 구현한 실험 프로젝트입니다.

중앙 Parameter Server와 다수의 Worker 노드가 협력하여 모델을 학습하는 구조로 구성되어 있습니다.

---

## 개요

본 프로젝트는 컨테이너 환경에서 분산 딥러닝 시스템의 구조와 동작 방식을 이해하고 실험하는 것을 목표로 합니다.

주요 특징은 다음과 같습니다.

* Parameter Server + Multi-Worker 구조
* PyTorch RPC 기반 통신
* 비동기 SGD 학습 방식
* Docker 기반 실행 환경
* GPU 지원 컨테이너

---

## 시스템 구조

```
+-------------------+
| Parameter Server  |
|   (rank = 0)      |
+-------------------+
         ↑
         | RPC
         ↓
+-------------------+     +-------------------+
|   Worker 1        | ... |   Worker N        |
| (rank = 1..N)     |     |                   |
+-------------------+     +-------------------+
```

* Parameter Server는 전역 모델 파라미터를 관리합니다.
* Worker는 개별 학습을 수행하고 gradient를 전송합니다.
* 모든 노드는 RPC 기반으로 통신합니다.

---

## 레포지토리 구조

```
.
├── docker_ddp.py        # 메인 학습 코드
├── report.pdf           # 실험 보고서
└── README.md
```

---

## 실행 방법

### 1. Docker 컨테이너 생성

#### Parameter Server

```bash
docker run --gpus all \
-it \
-p 6100:6100 \
--name ps \
-v /data:/home \
pytorch/pytorch \
/bin/bash
```

#### Worker 예시

```bash
docker run --gpus '"device=0"' \
-it \
-p 6101:6101 \
--name worker1 \
-v /data:/home \
pytorch/pytorch \
/bin/bash
```

---

### 2. Parameter Server 실행

```bash
python docker_ddp.py \
--rank 0 \
--world_size 3 \
--master_addr <PS_IP> \
--master_port 6100
```

---

### 3. Worker 실행

각 Worker 컨테이너에서 아래 명령어를 실행합니다.

```bash
python docker_ddp.py \
--rank 1 \
--world_size 3 \
--master_addr <PS_IP> \
--master_port 6100
```

```bash
python docker_ddp.py \
--rank 2 \
--world_size 3 \
--master_addr <PS_IP> \
--master_port 6100
```

---

## 주요 실행 옵션

| 옵션           | 설명      | 기본값               |
| ------------ | ------- | ----------------- |
| --model      | 모델 이름   | resnet18          |
| --rank       | 프로세스 순번 | 1                 |
| --world_size | 전체 노드 수 | 3                 |
| --data_dir   | 데이터 경로  | ./imagenette2/val |
| --batch_size | 배치 크기   | 256               |
| --lr         | 학습률     | 0.01              |
| --num_epochs | 학습 반복 수 | 150               |

---

## 구현 상세

### Parameter Server

* Worker로부터 gradient 수집
* SGD 기반 전역 파라미터 업데이트
* 업데이트된 모델 전송
* Thread Lock 기반 동기화
* 비동기 RPC 실행 지원

---

### GPU 할당 방식

Docker 환경에서는 컨테이너당 하나의 GPU만 인식되므로 device id를 0으로 고정합니다.

```python
device = torch.device("cuda:0")
```

이를 통해 GPU 충돌 문제를 방지합니다.

---

### 학습 흐름

각 Worker는 다음 순서로 학습을 수행합니다.

1. 미니배치 데이터 로딩
2. Forward 연산 수행
3. Loss 계산
4. Backward 연산 수행
5. Gradient 전송
6. 업데이트된 모델 수신

비동기 방식으로 동작하므로 Worker 간 대기 시간이 발생하지 않습니다.

---

## 출력 예시

학습 중 다음과 같은 로그가 출력됩니다.

```text
worker1 | Epoch: 1 | Batch: 3 | Loss: 1.42
Accuracy: 0.81
Time: 79.09 seconds
```


* 운영 환경을 고려한 최적화는 적용되지 않았습니다.
* 네트워크 환경에 따라 설정 변경이 필요할 수 있습니다.
