import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import precision_score, recall_score, f1_score
from types import SimpleNamespace
import numpy as np
import pandas as pd
from PIL import Image

import argparse
import yaml
import logging
from datetime import datetime

# CATS 모델 아키텍처 임포트
from CATS import Model as CatsDecoder

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
    logging.info("log 기록 시작.")

# =============================================================================
# 2. 이미지 인코더 모델 정의
# =============================================================================
class CnnFeatureExtractor(nn.Module):
    """
    다양한 CNN 아키텍처의 앞부분을 특징 추출기로 사용하는 범용 클래스입니다.
    run.yaml의 `cnn_feature_extractor.name` 설정에 따라 모델 구조가 결정됩니다.
    """
    def __init__(self, cnn_feature_extractor_name='resnet18_layer1', pretrained=True, in_channels=3, out_channels=None):
        super().__init__()
        self.cnn_feature_extractor_name = cnn_feature_extractor_name
        
        # CNN 모델 이름에 따라 모델과 잘라낼 레이어, 기본 출력 채널을 설정합니다.
        if cnn_feature_extractor_name == 'resnet18_layer1':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.front = nn.Sequential(*list(base_model.children())[:5]) # layer1까지
            base_out_channels = 64
        elif cnn_feature_extractor_name == 'resnet18_layer2':
            base_model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.front = nn.Sequential(*list(base_model.children())[:6]) # layer2까지
            base_out_channels = 128
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat1':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.front = base_model.features[:2] # features의 2번째 블록까지
            base_out_channels = 16
        elif cnn_feature_extractor_name == 'mobilenet_v3_small_feat3':
            base_model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.front = base_model.features[:4] # features의 4번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat2':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.front = base_model.features[:3] # features의 3번째 블록까지
            base_out_channels = 24
        elif cnn_feature_extractor_name == 'efficientnet_b0_feat3':
            base_model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
            self._adjust_input_channels(base_model, in_channels)
            self.front = base_model.features[:4] # features의 4번째 블록까지
            base_out_channels = 40
        else:
            raise ValueError(f"지원하지 않는 CNN 피처 추출기 이름입니다: {cnn_feature_extractor_name}")

        # 최종 출력 채널 수를 `featured_patch_channel`에 맞추기 위한 1x1 컨볼루션 레이어입니다.
        if out_channels is not None and out_channels != base_out_channels:
            self.channel_proj = nn.Conv2d(base_out_channels, out_channels, kernel_size=1)
        else:
            self.channel_proj = nn.Identity()

    def _adjust_input_channels(self, base_model, in_channels):
        """모델의 첫 번째 컨볼루션 레이어의 입력 채널을 조정합니다."""
        if in_channels == 1:
            # 첫 번째 conv 레이어 찾기
            if 'resnet' in self.cnn_feature_extractor_name:
                first_conv = base_model.conv1
                out_c, _, k, s, p, _, _, _ = first_conv.out_channels, first_conv.in_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, first_conv.dilation, first_conv.groups, first_conv.bias
                new_conv = nn.Conv2d(1, out_c, kernel_size=k, stride=s, padding=p, bias=False)
                with torch.no_grad():
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                base_model.conv1 = new_conv
            elif 'mobilenet' in self.cnn_feature_extractor_name or 'efficientnet' in self.cnn_feature_extractor_name:
                first_conv = base_model.features[0][0] # nn.Sequential -> Conv2dNormActivation -> Conv2d
                out_c, _, k, s, p, _, _, _ = first_conv.out_channels, first_conv.in_channels, first_conv.kernel_size, first_conv.stride, first_conv.padding, first_conv.dilation, first_conv.groups, first_conv.bias
                new_conv = nn.Conv2d(1, out_c, kernel_size=k, stride=s, padding=p, bias=False)
                with torch.no_grad():
                    new_conv.weight.copy_(first_conv.weight.mean(dim=1, keepdim=True))
                base_model.features[0][0] = new_conv
        elif in_channels != 3:
            raise ValueError("in_channels는 1 또는 3만 지원합니다.")

    def forward(self, x):
        x = self.front(x)
        x = self.channel_proj(x)
        return x

class PatchConvEncoder(nn.Module):
    """이미지를 패치로 나누고, 각 패치에서 특징을 추출하여 1D 시퀀스로 변환하는 인코더입니다."""
    def __init__(self, in_channels, img_size, patch_size, featured_patch_dim, cnn_feature_extractor_name):
        super(PatchConvEncoder, self).__init__()
        self.patch_size = patch_size
        self.featured_patch_dim = featured_patch_dim
        self.num_encoder_patches = (img_size // patch_size) ** 2
        
        self.shared_conv = nn.Sequential(
            CnnFeatureExtractor(cnn_feature_extractor_name=cnn_feature_extractor_name, pretrained=True, in_channels=in_channels, out_channels=featured_patch_dim),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(start_dim=1) # [B*N, D, 1, 1] -> [B*N, D] 형태가 됩니다.
        )
        self.norm = nn.LayerNorm(featured_patch_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        patches = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        # patches.shape: [B, C, num_patches_h, num_patches_w, patch_size, patch_size]
        patches = patches.permute(0, 2, 3, 1, 4, 5).reshape(-1, C, self.patch_size, self.patch_size)
        # patches.shape: [B * num_patches, C, patch_size, patch_size]
        
        conv_outs = self.shared_conv(patches)
        # 각 패치 특징 벡터에 대해 Layer Normalization 적용
        conv_outs = self.norm(conv_outs)
        # CATS 모델에 입력하기 위해 [B, num_patches, dim] 형태로 재구성
        conv_outs = conv_outs.view(B, self.num_encoder_patches, self.featured_patch_dim)
        return conv_outs

class Classifier(nn.Module):
    """CATS 모델의 출력을 받아 최종 클래스 로짓으로 매핑하는 분류기입니다."""
    def __init__(self, num_decoder_patches, featured_patch_dim, num_labels, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        # CATS 출력 특징 벡터를 최종 클래스 수로 매핑하는 선형 레이어
        self.projection = nn.Linear(num_decoder_patches * featured_patch_dim, num_labels)

    def forward(self, x):
        # x shape: [B, num_decoder_patches * featured_patch_dim]
        x = self.dropout(x)
        x = self.projection(x) # -> [B, num_labels]
        return x

class HybridModel(torch.nn.Module):
    """인코더와 CATS 분류기를 결합한 최종 하이브리드 모델입니다."""
    def __init__(self, encoder, decoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.classifier = classifier
        
    def forward(self, x):
        # 1. 인코딩: 2D 이미지 -> 패치 시퀀스
        x = self.encoder(x)
        # 2. 크로스-어텐션: 패치 시퀀스 -> 특징 벡터
        x = self.decoder(x)
        # 3. 분류: 특징 벡터 -> 클래스 로짓
        out = self.classifier(x)
        return out

# =============================================================================
# 3. 훈련 및 평가 함수
# =============================================================================
def log_model_parameters(model):
    """모델의 구간별 및 총 파라미터 수를 계산하고 로깅합니다."""
    
    def count_parameters(m):
        return sum(p.numel() for p in m.parameters() if p.requires_grad)

    encoder_params = count_parameters(model.encoder)
    decoder_params = count_parameters(model.decoder)
    classifier_params = count_parameters(model.classifier)
    total_params = encoder_params + decoder_params + classifier_params

    logging.info("="*50)
    logging.info("모델 파라미터 수:")
    logging.info(f"  - Encoder (PatchConvEncoder): {encoder_params:,} 개")
    logging.info(f"  - Decoder (CatsDecoder):      {decoder_params:,} 개")
    logging.info(f"  - Classifier (Linear Head):   {classifier_params:,} 개")
    logging.info(f"  - 총 파라미터:                  {total_params:,} 개")
    logging.info("="*50)

def evaluate(model, data_loader, device, desc="Evaluating"):
    """모델을 평가하고 정확도, 정밀도, 재현율, F1 점수를 로깅합니다."""
    model.eval()
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(data_loader, desc=desc, leave=False)
    with torch.no_grad():
        for images, labels in progress_bar:
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
    
    logging.info(f'{desc} | Accuracy: {accuracy:.2f}% | Precision: {precision:.4f} | Recall: {recall:.4f} | F1 Score: {f1:.4f}')
    return f1

def train(args, model, train_loader, valid_loader, device):
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
        
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs} [Training]", leave=False)
        for images, labels in progress_bar:
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
            
            # tqdm 프로그레스 바에 현재 loss 표시
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = 100 * correct / total
        logging.info(f'Epoch [{epoch+1}/{args.epochs}], Loss: {running_loss/len(train_loader):.4f}, Train Accuracy: {train_acc:.2f}%')
        
        # --- 평가 단계 ---
        f1 = evaluate(model, valid_loader, device, desc=f"Epoch {epoch+1}/{args.epochs} [Validation]")
        
        # 최고 성능 모델 저장
        if f1 > best_f1:
            best_f1 = f1
            torch.save(model.state_dict(), model_path)
            logging.info(f"최고 성능 모델 저장 완료 (F1 Score: {best_f1:.4f}) -> '{model_path}'")

def inference(args, model, data_loader, device, mode_name="추론"):
    """저장된 모델을 불러와 추론 시 GPU 메모리 사용량을 측정하고, 테스트셋 성능을 평가합니다."""
    logging.info(f"{mode_name} 모드를 시작합니다.")
    
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    model_path = os.path.join("checkpoints", data_dir_name, args.model_path)
    if not os.path.exists(model_path):
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 먼저 훈련을 실행하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"'{model_path}' 가중치 로드 완료.")
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
    evaluate(model, data_loader, device, desc=mode_name)

# =============================================================================
# 4. 데이터 준비 함수
# =============================================================================
class CustomImageDataset(Dataset):
    """CSV 파일과 이미지 폴더 경로를 받아 데이터를 로드하는 커스텀 데이터셋입니다."""
    def __init__(self, csv_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(csv_file)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_name = self.img_labels.iloc[idx, 0]
        img_path = os.path.join(self.img_dir, img_name)
        
        image = Image.open(img_path).convert("RGB")
        
        if self.transform:
            image = self.transform(image)

        label = int(self.img_labels.iloc[idx, 1])
        return image, label

def prepare_data(run_cfg, train_cfg, model_cfg, data_dir_name):
    """데이터셋을 로드하고 전처리하여 DataLoader를 생성합니다."""
    normalize = transforms.Normalize(mean=[0.5]*model_cfg.in_channels, std=[0.5]*model_cfg.in_channels)
    
    train_transform = transforms.Compose([
        transforms.Resize((model_cfg.img_size, model_cfg.img_size)),
        # 데이터 증강을 통해 모델 성능 향상 및 과적합 방지
        transforms.Resize((int(model_cfg.img_size*1.1), int(model_cfg.img_size*1.1))),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(model_cfg.img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.Grayscale(num_output_channels=model_cfg.in_channels),
        transforms.ToTensor(),
        normalize
    ])

    valid_test_transform = transforms.Compose([
        transforms.Resize((model_cfg.img_size, model_cfg.img_size)),
        transforms.Grayscale(num_output_channels=model_cfg.in_channels),
        transforms.ToTensor(),
        normalize
    ])
    
    try:
        logging.info("데이터 로드를 시작합니다.")

        # 커스텀 데이터셋 사용
        full_train_dataset = CustomImageDataset(csv_file=run_cfg.train_csv, img_dir=run_cfg.train_img_dir, transform=train_transform)
        full_valid_dataset = CustomImageDataset(csv_file=run_cfg.valid_csv, img_dir=run_cfg.valid_img_dir, transform=valid_test_transform)
        full_test_dataset = CustomImageDataset(csv_file=run_cfg.test_csv, img_dir=run_cfg.test_img_dir, transform=valid_test_transform)

        num_labels = 2 # 0과 1의 이진 분류
        class_names = ['Normal', 'Defect']

        # --- 데이터 샘플링 로직 ---
        sampling_ratio = getattr(run_cfg, 'sampling_ratio', 1.0)
        if sampling_ratio < 1.0:
            logging.info(f"데이터셋을 {sampling_ratio * 100:.0f}% 비율로 샘플링합니다 (random_seed={run_cfg.random_seed}).")
            
            def get_subset(dataset):
                """데이터셋에서 지정된 비율만큼 단순 랜덤 샘플링을 수행합니다."""
                num_total = len(dataset)
                num_to_sample = int(num_total * sampling_ratio)
                rng = np.random.default_rng(run_cfg.random_seed) # 재현성을 위한 랜덤 생성기
                indices = rng.choice(num_total, size=num_to_sample, replace=False)
                return Subset(dataset, indices)

            train_dataset = get_subset(full_train_dataset)
            valid_dataset = get_subset(full_valid_dataset)
            test_dataset = get_subset(full_test_dataset)
        else:
            logging.info("전체 데이터셋을 사용합니다 (sampling_ratio=1.0).")
            train_dataset = full_train_dataset
            valid_dataset = full_valid_dataset
            test_dataset = full_test_dataset

        # DataLoader 생성
        train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
        valid_loader = DataLoader(valid_dataset, batch_size=train_cfg.batch_size, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=train_cfg.batch_size, shuffle=False)
        
        logging.info(f"훈련 데이터: {len(train_dataset)}개, 검증 데이터: {len(valid_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
        
        return train_loader, valid_loader, test_loader, num_labels, class_names
        
    except FileNotFoundError:
        logging.error(f"데이터 폴더 또는 CSV 파일을 찾을 수 없습니다. 'run.yaml'의 경로 설정을 확인해주세요.")
        exit()


# =============================================================================
# 5. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    # --- YAML 설정 파일 로드 ---
    parser = argparse.ArgumentParser(description="YAML 설정을 이용한 CATS 기반 이미지 분류기")
    parser.add_argument('--config', type=str, default='run.yaml', help='설정 파일 경로')
    args = parser.parse_args()

    with open(args.config, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    # SimpleNamespace를 사용하여 딕셔너리처럼 접근 가능하게 변환
    run_cfg = SimpleNamespace(**config['run'])
    train_cfg = SimpleNamespace(**config['training'])
    model_cfg = SimpleNamespace(**config['model'])
    cats_cfg = SimpleNamespace(**model_cfg.cats)

    # 로그 폴더 이름을 'Sewer-ML'과 같이 명시적으로 지정
    setup_logging("Sewer-ML")
    
    # --- 설정 파일 내용 로깅 ---
    config_str = yaml.dump(config, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logging.info("="*50)
    logging.info("run.yaml:")
    logging.info("\n" + config_str)
    logging.info("="*50)
    
    # --- 공통 파라미터 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 데이터 준비 ---
    train_loader, valid_loader, test_loader, num_labels, class_names = prepare_data(run_cfg, train_cfg, model_cfg, "Sewer-ML")

    # --- 모델 구성 ---
    num_encoder_patches = (model_cfg.img_size // model_cfg.patch_size) ** 2 # 16
    
    cats_params = {
        'num_encoder_patches': num_encoder_patches,
        'num_labels': num_labels, 'd_layers': cats_cfg.d_layers,
        'emb_dim': cats_cfg.emb_dim,
        'd_ff': cats_cfg.emb_dim * cats_cfg.d_ff_ratio,
        'n_heads': cats_cfg.n_heads,
        'featured_patch_dim': cats_cfg.featured_patch_dim,
        'dropout': cats_cfg.dropout,
        'positional_encoding': cats_cfg.positional_encoding,
        'store_attn': cats_cfg.store_attn,
        'QAM_start': cats_cfg.qam['start'],
        'QAM_end': cats_cfg.qam['end'],
    }
    cats_args = SimpleNamespace(**cats_params)

    # cli_args 대신 설정 파일 값들을 전달 (추론 시 사용)
    cli_args = SimpleNamespace(
        mode=run_cfg.mode, 
        # data_dir은 이제 여러 개이므로 대표 이름으로 설정
        data_dir="Sewer-ML", 
        batch_size=train_cfg.batch_size,
        epochs=train_cfg.epochs, lr=train_cfg.lr, model_path=run_cfg.model_path,
        img_size=model_cfg.img_size, in_channels=model_cfg.in_channels
    )

    encoder = PatchConvEncoder(in_channels=model_cfg.in_channels, img_size=model_cfg.img_size, patch_size=model_cfg.patch_size, 
                               featured_patch_dim=cats_cfg.featured_patch_dim, cnn_feature_extractor_name=model_cfg.cnn_feature_extractor['name'])
    decoder = CatsDecoder(args=cats_args) # CATS.py의 Model 클래스
    
    num_decoder_patches = (num_labels + cats_cfg.featured_patch_dim - 1) // cats_cfg.featured_patch_dim
    classifier = Classifier(num_decoder_patches=num_decoder_patches, 
                            featured_patch_dim=cats_cfg.featured_patch_dim, num_labels=num_labels, dropout=cats_cfg.dropout)
    model = HybridModel(encoder, decoder, classifier).to(device)

    # 모델 생성 후 파라미터 수 로깅
    log_model_parameters(model)
    
    # --- 모드에 따라 실행 ---
    if run_cfg.mode == 'train':
        # 훈련 시에는 train_loader와 valid_loader 사용
        train(cli_args, model, train_loader, valid_loader, device)
        
        logging.info("="*50)
        logging.info("훈련 완료. 최종 모델 성능을 테스트 세트로 평가합니다.")
        # 훈련 종료 후, 최종 평가를 위해 test_loader 사용.
        inference(cli_args, model, test_loader, device, mode_name="Final Evaluation")
    elif run_cfg.mode == 'inference':
        # 추론 모드에서는 test_loader를 사용해 성능 평가
        inference(cli_args, model, test_loader, device, mode_name="Inference")