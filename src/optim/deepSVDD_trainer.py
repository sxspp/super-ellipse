from base.base_trainer import BaseTrainer
from base.base_dataset import BaseADDataset
from base.base_net import BaseNet
from torch.utils.data.dataloader import DataLoader
from sklearn.metrics import roc_auc_score
from optim.ae_trainer import visualize_tsne2
import logging
import time
import torch
import torch.optim as optim
import numpy as np
import cv2

import matplotlib.pyplot as plt

class DeepSVDDTrainer(BaseTrainer):

    def __init__(self, objective, R, c, nu: float, optimizer_name: str = 'adam', lr: float = 0.001, n_epochs: int = 150,
                 lr_milestones: tuple = (), batch_size: int = 128, weight_decay: float = 1e-6, device: str = 'cuda',
                 n_jobs_dataloader: int = 0):
        super().__init__(optimizer_name, lr, n_epochs, lr_milestones, batch_size, weight_decay, device,
                         n_jobs_dataloader)

        assert objective in ('one-class', 'soft-boundary'), "Objective must be either 'one-class' or 'soft-boundary'."
        self.objective = objective

        # Deep SVDD parameters
        self.R = torch.tensor(R, device=self.device)  # 반지름은 초기값은 0 으로 주어지고 업데이트됨 
        self.c = torch.tensor(c, device=self.device) if c is not None else None #
        self.nu = nu

        # Optimization parameters
        self.warm_up_n_epochs = 5  # number of training epochs for soft-boundary Deep SVDD before radius R gets updated

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()
        global c1v,c2v,a
        # Set device for network
        net = net.to(self.device)

        # Get train data loader
        train_loader, _ = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Set optimizer (Adam optimizer for now)
        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay,
                               amsgrad=self.optimizer_name == 'amsgrad')

        # Set learning rate scheduler
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=self.lr_milestones, gamma=0.1)
####sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp###
        # Initialize hypersphere center c (if c not loaded) 초기센터 C 설정하는 곳
        if self.c is None:
            logger.info('Initializing center c...')
            self.c = self.init_center_c(train_loader, net) #
            logger.info('Center c initialized.')

        # Training
        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            scheduler.step()
            if epoch in self.lr_milestones:
                logger.info('  LR scheduler: new learning rate is %g' % float(scheduler.get_lr()[0]))

            loss_epoch = 0.0
            n_batches = 0
            epoch_start_time = time.time()
            for data in train_loader:
                inputs, _, _ = data
                inputs = inputs.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()
####sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp#######sxspp###
#               c는 센터
#               R은 c1c2 구하기 위한 원의 반지름
#               nu는 V (하이퍼파라미터 값으로 0에서 1사이의 값을 가지고 구의 부피와 경계선의 위반 사이의 균형을 제어합니다.)
                #Update network parameters via backpropagation: forward + backward + optimize
            #     outputs = net(inputs)
            #     dist = torch.sum((outputs - self.c) ** 2, dim=1)    # outputs과 센터c와의 거리를 계산한 열 들의 합
            #    # print(dist.max())
            #     if self.objective == 'soft-boundary': #deepSVDD (3)식 = 냀-boundary
            #         scores = dist - self.R ** 2 #scores = outputs - c 의 **2 - R **2
            #         loss = self.R ** 2 + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)) #두번째 항은 중심에서 R보다 먼 값에는 패널티 부여하는 식 
            #     else: #objective가 one-class일 경우 
            #         loss = torch.mean(dist)     #deepSVDD (4)의 식 
            #     loss.backward() # weight 업데이트 
            #     optimizer.step() #최적화
##############################################아래부분 수정부분###############################################################################
                outputs = net(inputs) #inputs을 net태워서 output으로 출력
                dist = torch.sum((outputs - self.c)**2, dim=1)    # outputs과 센터c와의 거리를 계산한 열 들의 합
                a = dist.max() #센터C로 부터 가장 멀리있는 output의 거리 ///타원의 장축
                b = 0 # 장축과 센터 C로 부터의 반지름R 중간값 ///타원의 단축
                c1 = torch.sqrt(abs(a - b**2))# C1**2 = A**2 - B**2을 사용 /// 타원의 센터C에서 초점 C1까지의 거리
                
                arg = torch.argmax(dist) #가장 멀리 떨어져있는 거리 인덱스 값 가져옴
                av = outputs[arg] # 인덱스 값으로 output값 뽑아오기
                
                cav = av-self.c  #C->A 벡터 계산
                uv = cav/torch.sqrt((cav**2)) #C->A 벡터 제곱해서 루트 씌우고 분모에 넣음 단위벡터 생성
                c1v =  self.c+(uv*c1) #c->c1 까지 벡터 단위벡터에 c1 곱해서 생성
                c2v =  self.c-(uv*c1) # 단위벡터에 - 를 해서 c2 벡터 생성

                newdist = torch.sum(((outputs - self.c) + (self.c - c1v)) + ((outputs - self.c)+(self.c - c2v)), dim=1)
                #outputs이 c1,c2와 거리합이 2a인 타원의 값을 이용해서 newdist 저장  newdist 자체가 score!!!

                if self.objective == 'soft-boundary': #deepSVDD (3)식 = 냀-boundary
                    scores = newdist - 2*a #scores 를 타원의 정의를 이용해서 score 계산
                    loss = 2*a + (1 / self.nu) * torch.mean(torch.max(torch.zeros_like(scores), scores)) #두번째 항은 중심에서 R보다 먼 값에는 패널티 부여하는 식 
                else: #objective가 one-class일 경우 
                    loss = torch.mean(newdist-(2*a))     #deepSVDD (4)의 식 
                loss.backward() # weight 업데이트 
                optimizer.step() #최적화
################################################윗부분 수정중############################################################################
                # Update hypersphere radius R on mini-batch distances
                if (self.objective == 'soft-boundary') and (epoch >= self.warm_up_n_epochs): #objective가 soft-boundary이고  에폭이 웜업을 넘었을때
                    b = torch.tensor(get_c1c2(dist, self.nu), device=self.device) #C1,C2 반지름 R을 업데이트 합니다.
                    #self.R.data를 b로 변경
                loss_epoch += loss.item() #loss tensor에서 값만 가져와서 loss_epoch 증가
                n_batches += 1 #배치 카운트 증가

            # log epoch statistics
            epoch_train_time = time.time() - epoch_start_time
            logger.info('  Epoch {}/{}\t Time: {:.3f}\t Loss: {:.8f}'
                        .format(epoch + 1, self.n_epochs, epoch_train_time, loss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net
    def test(self, dataset: BaseADDataset, net: BaseNet):
        logger = logging.getLogger()

        # Set device for network
        net = net.to(self.device)

        # Get test data loader
        _, test_loader = dataset.loaders(batch_size=self.batch_size, num_workers=self.n_jobs_dataloader)

        # Testing
        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for data in test_loader:
                inputs, labels, idx = data
                inputs = inputs.to(self.device)
                outputs = net(inputs)
                newdist = torch.sum(((outputs - self.c) + (self.c - c1v)) + ((outputs - self.c)+(self.c - c2v)), dim=1)
                if self.objective == 'soft-boundary':
                    scores = newdist - 2*a
                else:
                    scores = newdist

                # Save triples of (idx, label, score) in a list
                idx_label_score += list(zip(idx.cpu().data.numpy().tolist(),
                                            labels.cpu().data.numpy().tolist(),
                                            scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        # Compute AUC
        _, labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        logger.info('Test set AUC: {:.2f}%'.format(100. * self.test_auc))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader: DataLoader, net: BaseNet, eps=0.1): #초기 센터 C 설정
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        c = torch.zeros(net.rep_dim, device=self.device) # 센터 C 저장하는 0행렬 제작

        net.eval()
        with torch.no_grad():
            for data in train_loader:
                #print(data)
                # get the inputs of the batch
                inputs, _, _ = data #1번째 데이터만 input으로 받고 나머지 버림
                inputs = inputs.to(self.device) #장치 사용
                outputs = net(inputs) # NN지난 출력값을 output으로 사용
                n_samples += outputs.shape[0] #batch수를 n_samples에 저장
                c += torch.sum(outputs, dim=0) #outputs에 행의 값을 더해서 c에 더함
                #visualize_tsne2(outputs)  #라벨값 올바른 변수 입력 요망

        c /= n_samples # C를 총 sample 수로 나눈 값이 C로 업데이트

        # If c_i is too close to 0, set to +-eps. Reason: a zero unit can be trivially matched with zero weights.
        c[(abs(c) < eps) & (c < 0)] = -eps
        c[(abs(c) < eps) & (c > 0)] = eps

        return c

            #b를 newdist에서 변경
def get_c1c2(b: torch.Tensor, nu: float): #c1,c2를 구하기위한 원
    """Optimally solve for radius R via the (1-nu)-quantile of distances."""
    return np.quantile(np.sqrt(b.clone().data.cpu().numpy()), 1 - nu)
                                #b를 newdist에서 변경
