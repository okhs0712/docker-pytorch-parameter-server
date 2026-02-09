#device id 부분 수정 : 컨테이너의 GPU는 다 1개니까.

import argparse
import os
import threading
import time

import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.distributed.rpc as rpc
from torchvision import transforms, datasets, models
import matplotlib.pyplot as plt

model_dict = {'resnet18': models.resnet18, 'resnet50': models.resnet50, 'vgg16': models.vgg16, 'alexnet': models.alexnet,
              'googlenet': models.googlenet, 'inception': models.inception_v3,
              'densenet121': models.densenet121, 'mobilenet': models.mobilenet_v2}

class ParameterServer(object):
    """"
     The parameter server (PS) updates model parameters with gradients from the workers
     and sends the updated parameters back to the workers.
    """
    def __init__(self, model, num_workers, lr): #파라미터 서버 초기화
        self.lock = threading.Lock() 
        ''' 쓰레딩 락 : lock을 acquire 시, 해당 쓰레드만 공유 데이터에 접근 가능, 
        lock을 release 해야 다른 쓰레드에서 공유 가능.'''
        self.future_model = torch.futures.Future()
        self.num_workers = num_workers
        # initialize model parameters
        assert model in model_dict.keys(), \
            f'model {model} is not in the model list: {list(model_dict.keys())}'
        self.model = model_dict[model](num_classes=10)
        # zero gradients
        for p in self.model.parameters(): # 모델의 모든 매개변수에 대한 그레디언트 초기화
            p.grad = torch.zeros_like(p)
        self.optimizer = optim.SGD(self.model.parameters(), lr=lr, momentum=0.9) # SGD 사용

    def get_model(self): # 모델 반환
        return self.model

    @staticmethod
    @rpc.functions.async_execution # async로 동작
    def update_and_fetch_model(ps_rref, grads, worker_rank): # 매개변수 업데이트 및 결과 반환
        self = ps_rref.local_value()
        with self.lock:
            print(f'PS updates parameters based on gradients from worker{worker_rank}')
            # update model parameters
            for p, g in zip(self.model.parameters(), grads): # worker한테 받은 그레디언트 업데이트
                p.grad = g
            self.optimizer.step() 
            self.optimizer.zero_grad() # 다음 iter를 위해 결과 초기화

            fut = self.future_model # 결과 저장

            fut.set_result(self.model) 
            self.future_model = torch.futures.Future() # 다음 iter를 위해 결과 초기화

        return fut

loss_list = []
epoch_list = []
accuracy_list = []

def run_worker(ps_rref, rank, data_dir, batch_size, num_epochs): # 함수의 인자에 device_id_set 추가.
    """
    A worker pulls model parameters from the PS, computes gradients on a mini-batch
    from its data partition, and pushes the gradients to the PS.
    """
    worker_num = rank
    # prepare dataset
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) # 정규화
    transform = transforms.Compose( # transform
        [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip(), transforms.ToTensor(), normalize])
    train_dataset = datasets.ImageFolder(root=data_dir, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, ) 

    #set device
    #device_id = rank - 1 

    # docker에서 컨테이너마다 gpu 1개로 돌리므로 device id 수정.
    # device id는 머신마다 다름. 따라서 기존의 'rank - 1'에서  수정.
    if rank==1: 
        device_id = 0
    elif rank==2 : 
        device_id = 0
    elif rank==3 : 
        device_id = 0
    elif rank==4 : 
        device_id = 0
    
    # device_id = 0
    device = torch.device(f"cuda:{device_id}" if torch.cuda.is_available() else "cpu")
    criterion = nn.CrossEntropyLoss()

    # get initial model from the PS
    m = ps_rref.rpc_sync().get_model().to(device) # worker device로 모델 전송

    print(f'worker{rank} starts training')
    tt0 = time.time()

    for i in range(num_epochs):
        correct_sum = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            start_time = time.time() # 함수 실행 이전 시간 측정
            data, target = data.to(device), target.to(device)
            output = m(data)
            loss = criterion(output, target)
            loss.backward()

            pred = output.argmax(dim=1, keepdim=True) # accuracy calculate
            pred, target = pred.to(device), target.to(device)
            correct = pred.eq(target.view_as(pred)).sum().item()
            correct_sum += correct
            end_time = time.time() # 함수 실행 이후 시간 측정


            print("worker{:d} | Epoch:{:3d} | Batch: {:3d} | Loss: {:6.2f}"
                  .format(rank, (i + 1), (batch_idx + 1), loss.item()))
            elapsed_time = end_time - start_time # 실행 시간 계산
            print(f"elapsed_time : {elapsed_time:.6f} sec") # 소수점 아래 6자리까지 출력


            # send gradients to the PS and fetch updated model parameters
            m = rpc.rpc_sync(to=ps_rref.owner(),
                             func=ParameterServer.update_and_fetch_model,
                             args=(ps_rref, [p.grad for p in m.cpu().parameters()], rank)
                             ).to(device)
        loss_list.append(loss.item())
        epoch_list.append(i) # epoch 몇 인지 list에 저장.
        accuracy_list.append(correct_sum / len(train_loader.dataset))
        print("Accuracy: {:6.2f}".format(correct_sum / len(train_loader.dataset)))
    tt1 = time.time()

    print("Time: {:.2f} seconds".format((tt1 - tt0)))


def main():
    parser = argparse.ArgumentParser(description="Train models on Imagenette under ASGD")
    parser.add_argument("--model", type=str, default="resnet18", help="The job's name.")
    parser.add_argument("--rank", type=int, default=1, help="Global rank of this process.")
    parser.add_argument("--world_size", type=int, default=3, help="Total number of workers.")
    parser.add_argument("--data_dir", type=str, default="./imagenette2/val", help="The location of dataset.")
    parser.add_argument("--master_addr", type=str, default="220.67.133.165", help="Address of master.")
    parser.add_argument("--master_port", type=str, default="6100", help="Port that master is listening on.")
    parser.add_argument("--batch_size", type=int, default=256, help="Batch size of each worker during training.")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate.")
    parser.add_argument("--num_epochs", type=int, default=150, help="Number of epochs.")
    #parser.add_argument("--device_id_set", type=int, default=0, help="여러 머신에서 코드를 돌리면 device id가 머신마다 다름")

    args = parser.parse_args()
    
    os.environ['MASTER_ADDR'] = args.master_addr
    os.environ['MASTER_PORT'] = args.master_port
    
    os.environ['NCCL_SOCKET_IFNAME'] = 'enp8s0'
    os.environ['GLOO_SOCKET_IFNAME'] = 'enp8s0'
    os.environ['TP_SOCKET_IFNAME'] = 'enp8s0'
    
    worker_num = args.rank

    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=16, rpc_timeout=800000)

    if args.rank == 0: # rank가 0이면 PS
        """
        initialize PS and run workers
        """
        print(f"PS{args.rank} initializing")
        rpc.init_rpc(f"PS{args.rank}", rank=args.rank, world_size=args.world_size, rpc_backend_options=options) #rpc 프레임워크 초기화
        print(f"PS{args.rank} initialized")

        ps_rref = rpc.RRef(ParameterServer(args.model, args.world_size, args.lr))

        futs = []
        for r in range(1, args.world_size):
            worker = f'worker{r}'
            futs.append(rpc.rpc_async(to=worker,
                                      func=run_worker,
                                      args=(ps_rref, r, args.data_dir, args.batch_size, args.num_epochs)))
        # python public-asgd_sync.py --rank=0 --world_size=3 --master_addr=220.67.133.165

        torch.futures.wait_all(futs)
        print(f"Finish training")

    else: # rank가 0이 아니면 worker
        """
        initialize workers
        """
        print(f"worker{args.rank} initializing")
        rpc.init_rpc(f"worker{args.rank}", rank=args.rank, world_size=args.world_size, rpc_backend_options=options)
        print(f"worker{args.rank} initialized")

    rpc.shutdown()

if __name__ == "__main__":
    main()
    
