import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, models, transforms
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from types import SimpleNamespace
import numpy as np
import argparse
import logging
from datetime import datetime

# CATS 모델 아키텍처 임포트
from CATS import Model as CATS_Model

# =============================================================================
# 1. 로깅 설정
# =============================================================================
def setup_logging(data_dir):
    """로그 파일을 log 폴더에 생성하고, 콘솔에도 함께 출력하도록 설정합니다."""
    data_dir_name = os.path.basename(os.path.normpath(data_dir))
    log_dir = os.path.join("log", data_dir_name)
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(log_dir, f"log_{timestamp}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_filename),
            logging.StreamHandler()
        ]
    )
    logging.info("로깅을 시작합니다.")

# =============================================================================
# 2. 이미지 인코더 모델 정의 (ResNetFront, PatchConvEncoder)
# =============================================================================
class ResNetFront(nn.Module):
    """사전 훈련된 ResNet의 앞부분을 특징 추출기로 사용하는 클래스입니다."""
    def __init__(self, backbone='resnet50', pretrained=True, in_channels=3, out_channels=None):
        super().__init__()
        # PyTorch 1.13 이상에서는 `weights` 파라미터를 권장합니다.
        if pretrained:
            resnet = getattr(models, backbone)(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = getattr(models, backbone)(weights=None)

        if in_channels == 1:
            conv1_weight = resnet.conv1.weight
            new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
            with torch.no_grad():
                new_conv1.weight.copy_(conv1_weight.mean(dim=1, keepdim=True))
            resnet.conv1 = new_conv1
        elif in_channels != 3:
            raise ValueError("in_channels는 1 또는 3만 지원합니다.")

        self.front = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu,
            resnet.maxpool, resnet.layer1,
        )

        self.base_out_channels = 256 if '50' in backbone else 64

        if out_channels is not None and out_channels != self.base_out_channels:
            self.channel_proj = nn.Conv2d(self.base_out_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()

    def forward(self, x):
        x = self.front(x)
        x = self.channel_proj(x)
        return x

class PatchConvEncoder(nn.Module):
    """이미지를 패치로 나누고, 각 패치에서 특징을 추출하여 1D 시퀀스로 변환하는 인코더입니다."""
    def __init__(self, in_channels, img_size, patch_size, hidden_dim, output_dim=None):
        super(PatchConvEncoder, self).__init__()
        self.patch_size = patch_size
        self.hidden_dim = hidden_dim
        self.num_patches = (img_size // patch_size) ** 2
        
        self.shared_conv = nn.Sequential(
            ResNetFront(backbone='resnet50', pretrained=True, in_channels=in_channels, out_channels=hidden_dim),
            nn.AdaptiveAvgPool2d((1, 1))
        )

        if output_dim is not None:
            self.proj = nn.Linear(self.num_patches * hidden_dim, output_dim)
        else:
            self.proj = None

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, self.patch_size, self.patch_size)
        
        conv_outs = self.shared_conv(patches)
        conv_outs = conv_outs.view(conv_outs.size(0), -1)
        
        conv_outs = conv_outs.view(B, self.num_patches * self.hidden_dim)
        
        if self.proj is not None:
            conv_outs = self.proj(conv_outs)
            
        return conv_outs

class XModel(torch.nn.Module):
    """인코더와 CATS 분류기를 결합한 최종 하이브리드 모델입니다."""
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        
    def forward(self, x):
        # 1. 인코딩: 2D 이미지 -> 1D 시퀀스
        x = self.encoder(x).unsqueeze(-1)
        # 2. 분류: 1D 시퀀스 -> 클래스 로짓
        out = self.classifier(x).squeeze(-1)
        return out

# =============================================================================
# 3. 훈련 및 평가 함수
# =============================================================================
def evaluate(model, test_loader, device):
    """모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    if total == 0:
        logging.warning("테스트 데이터가 없습니다. 평가를 건너뜁니다.")
        return 0.0

    accuracy = 100 * correct / total
    precision = precision_score(all_labels, all_preds, average='macro', zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro', zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    
    logging.info(f'Test Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')
    return f1

def train(args, model, train_loader, test_loader, device):
    """모델 훈련 및 평가를 수행하고 최고 성능 모델을 저장합니다."""
    logging.info("훈련 모드를 시작합니다.")
    
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    checkpoints_dir = os.path.join("checkpoints", data_dir_name)
    os.makedirs(checkpoints_dir, exist_ok=True)
    model_path = os.path.join(checkpoints_dir, args.model_path)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    best_f1 = 0.0

    for epoch in range(args.epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_acc = 100 * correct / total
        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%')
        
        # --- 평가 단계 ---
        f1 = evaluate(model, test_loader, device)
        
        # 최고 성능 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            logging.info(f"최고 성능 모델 저장 완료 (F1 Score: {best_f1:.4f}) -> '{model_path}'")

def inference(args, model, test_loader, device):
    """저장된 모델을 불러와 추론 시 GPU 메모리 사용량을 측정하고, 테스트셋 성능을 평가합니다."""
    logging.info("추론 모드를 시작합니다.")
    
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    model_path = os.path.join("checkpoints", data_dir_name, args.model_path)
    if not os.path.exists(model_path):
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 먼저 훈련을 실행하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"'{model_path}'에서 모델 가중치를 성공적으로 불러왔습니다.")
    except Exception as e:
        logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    model.eval()
    
    # 1. GPU 메모리 사용량 측정
    dummy_input = torch.randn(1, args.in_channels, args.img_size, args.img_size).to(device)
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
    
    with torch.no_grad():
        output = model(dummy_input)
        
    if torch.cuda.is_available():
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"추론 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 GPU 메모리 사용량을 측정할 수 없습니다.")

    # 2. 테스트셋 성능 평가
    logging.info("테스트셋에 대한 성능 평가를 시작합니다.")
    evaluate(model, test_loader, device)

# =============================================================================
# 4. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CATS 기반 이미지 분류기")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='실행 모드 (train 또는 inference)')
    parser.add_argument('--data_dir', type=str, default='./Refined_mix', help='데이터셋 폴더 경로')
    parser.add_argument('--batch_size', type=int, default=16, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='총 에포크 수')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='저장/로드할 모델 파일 이름')
    
    cli_args = parser.parse_args()
    
    setup_logging(cli_args.data_dir)
    
    # --- 공통 파라미터 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = 480
    in_channels = 1
    
    cli_args.img_size = img_size
    cli_args.in_channels = in_channels

    # --- 데이터 준비 ---
    normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    
    train_transform = transforms.Compose([
        transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=in_channels),
        transforms.ToTensor(),
        normalize
    ])

    test_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.Grayscale(num_output_channels=in_channels),
        transforms.ToTensor(),
        normalize
    ])
    
    try:
        base_dataset = datasets.ImageFolder(root=cli_args.data_dir)
        targets = base_dataset.targets
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(range(len(targets)), targets))

        train_dataset = Subset(datasets.ImageFolder(root=cli_args.data_dir, transform=train_transform), train_idx)
        test_dataset = Subset(datasets.ImageFolder(root=cli_args.data_dir, transform=test_transform), test_idx)

        train_loader = DataLoader(train_dataset, batch_size=cli_args.batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False)
        logging.info(f"데이터 로딩 완료. 훈련 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
    except FileNotFoundError:
        logging.error(f"데이터 폴더 '{cli_args.data_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    # --- 모델 구성 ---
    patch_len = 32
    emb_dim = 24
    n_heads = 4
    d_layers = 2
    patch_size = 120
    num_labels = len(base_dataset.classes)
    
    patch_num = (img_size // patch_size) ** 2
    seq_len = patch_num * patch_len
    
    cats_params = {
        'seq_len': seq_len, 'pred_len': num_labels, 'd_layers': d_layers,
        'dec_in': 1, 'd_model': emb_dim, 'd_ff': emb_dim * 2, 'n_heads': n_heads,
        'patch_len': patch_len, 'stride': patch_len, 'classification': True,
        'dropout': 0.1, 'query_independence': False, 'padding_patch': 'end',
        'store_attn': False, 'QAM_end': 0.0, 'QAM_start': 0.0, 'features': 'S',
        'des': 'Exp', 'itr': 1,
    }
    cats_args = SimpleNamespace(**cats_params)

    encoder = PatchConvEncoder(in_channels=in_channels, img_size=img_size, patch_size=patch_size, hidden_dim=patch_len)
    classifier = CATS_Model(args=cats_args)
    model = XModel(encoder, classifier).to(device)
    
    # --- 모드에 따라 실행 ---
    if cli_args.mode == 'train':
        train(cli_args, model, train_loader, test_loader, device)
    elif cli_args.mode == 'inference':
        inference(cli_args, model, test_loader, device)
