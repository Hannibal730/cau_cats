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
import matplotlib.pyplot as plt
import seaborn as sns

# CATS 모델 아키텍처 임포트
from CATS import Model as CATS_Model

# =============================================================================
# 1. 로깅 설정 (동일)
# ... (이전과 동일한 setup_logging 함수) ...
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
# 2. 이미지 인코더 및 하이브리드 모델 정의 (2개의 XModel 정의)
# =============================================================================
class ResNetFront(nn.Module):
    # ... (이전과 동일) ...
    def __init__(self, backbone='resnet50', pretrained=True, in_channels=3, out_channels=None):
        super().__init__()
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
    # ... (이전과 동일) ...
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


# ★★★ 1채널(흑백)용 표준 모델 ★★★
class XModel(torch.nn.Module):
    """인코더와 CATS 분류기를 결합한 표준 하이브리드 모델입니다."""
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder
        self.classifier = classifier
        
    def forward(self, x):
        x = self.encoder(x).unsqueeze(-1)
        out = self.classifier(x).squeeze(-1)
        return out

# ★★★ 3채널(RGB) 채널 독립성 모델 ★★★
class XModel_ChannelIndependent(torch.nn.Module):
    """각 채널을 이어붙여 처리하는 하이브리드 모델입니다."""
    def __init__(self, encoder, classifier):
        super().__init__()
        self.encoder = encoder # 인코더는 1채널 입력을 받도록 설계되어야 함
        self.classifier = classifier
        
    def forward(self, x):
        batch_size = x.shape[0]

        # 1단계: 3채널 이미지를 각 채널별로 분할하고,배치 사이즈를 3배 늘림
        # [B, 3, H, W] -> [B*3, 1, H, W]
        # 이유: 1채널용 인코더를 공유하여 모든 채널을 효율적으로 한 번에 처리하기 위함
        x_channels = x.view(batch_size * 3, 1, x.shape[2], x.shape[3])
        
        # 2단계: 레즈넷 기반 인코더로 각 채널별 특징 추출
        # [B*3, 1, H, W] -> [B*3, seq_len_single]
        # 이유: 2D 이미지 정보를 1D 시퀀스 정보로 변환
        channel_sequences = self.encoder(x_channels)
        
        # 3단계: 각 채널별 특징 시퀀스를 하나로 결합. 배치 사이즈도 원상복귀
        # [B*3, seq_len_single] -> [B, 3 * seq_len_single]
        # 이유: 최종 분류기가 R,G,B 모든 채널 정보를 종합하여 채널 간 관계를 학습하도록 함
        combined_sequence = channel_sequences.view(batch_size, -1)
        
        # 4단계: CATS 입력 및 최종 출력
        # [B, 3*seq_len_single] -> [B, num_classes]
        final_sequence = combined_sequence.unsqueeze(-1)
        out = self.classifier(final_sequence).squeeze(-1)
        
        return out

# =============================================================================
# 3. 훈련 및 평가 함수 (동일)
# ... (이전과 동일한 evaluate, train, inference 함수) ...
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

def inference(args, model, test_loader, device, class_names):
    """저장된 모델을 불러와 추론을 수행하고, 필요 시 어텐션 맵을 저장합니다."""
    logging.info("추론 모드를 시작합니다.")
    
    data_dir_name = os.path.basename(os.path.normpath(args.data_dir))
    model_path = os.path.join("checkpoints", data_dir_name, args.model_path)
    if not os.path.exists(model_path):
        logging.error(f"모델 파일('{model_path}')을 찾을 수 없습니다. 먼저 훈련을 실행하세요.")
        return

    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        logging.info(f"모델 가중치 로딩 완료. '{model_path}'")
    except Exception as e:
        logging.error(f"모델 가중치 로딩 중 오류 발생: {e}")
        return

    # --- 파라미터 수 상세 분석 ---
    def count_params(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    encoder_params = count_params(model.encoder)
    classifier_params = count_params(model.classifier)
    cats_model = model.classifier.model
    dummy_embedding = cats_model.backbone
    embedding_params = count_params(dummy_embedding.W_P) + (dummy_embedding.PE.numel() if dummy_embedding.PE.requires_grad else 0) + (dummy_embedding.dummies.numel() if dummy_embedding.dummies.requires_grad else 0)
    decoder_params = count_params(dummy_embedding.decoder)
    projection_params = count_params(cats_model.proj)
    total_params = count_params(model)

    logging.info("===== 모델 파라미터 상세 분석 =====")
    logging.info(f"- Encoder (Feature Extractor) : {encoder_params:,}")
    logging.info(f"- Classifier (Transformer)    : {classifier_params:,}")
    logging.info(f"    - Embedding               : {embedding_params:,}")
    logging.info(f"    - Decoder Layers          : {decoder_params:,}")
    logging.info(f"    - Projection Head         : {projection_params:,}")
    logging.info("-----------------------------------")
    logging.info(f"총 학습 가능한 파라미터 수      : {total_params:,}")
    logging.info("===================================")

    model.eval()

    # --- GPU 메모리 사용량 측정 ---
    if torch.cuda.is_available():
        dummy_input = torch.randn(1, args.in_channels, args.img_size, args.img_size).to(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            _ = model(dummy_input)
        peak_memory_bytes = torch.cuda.max_memory_allocated(device)
        peak_memory_mb = peak_memory_bytes / (1024 * 1024)
        logging.info(f"추론 시 최대 GPU 메모리 사용량: {peak_memory_mb:.2f} MB")
    else:
        logging.info("CUDA를 사용할 수 없어 GPU 메모리 사용량을 측정할 수 없습니다.")

    if args.save_attention_maps:
        attention_map_dir = os.path.join("attention_map", data_dir_name)
        os.makedirs(attention_map_dir, exist_ok=True)
        logging.info(f"어텐션 맵을 '{attention_map_dir}' 폴더에 저장합니다.")
        
        patch_num_per_channel = (args.img_size // args.patch_size) ** 2
        if args.in_channels == 3:
            r_end = patch_num_per_channel
            g_end = 2 * patch_num_per_channel
            b_end = 3 * patch_num_per_channel
            range_text = f"X-axis: R[0-{r_end-1}], G[{r_end}-{g_end-1}], B[{g_end}-{b_end-1}], Padding[{b_end}]"
        else:
            range_text = f"X-axis: Features[0-{patch_num_per_channel-1}], Padding[{patch_num_per_channel}]"

    with torch.no_grad():
        test_indices = test_loader.dataset.indices
        original_samples = test_loader.dataset.dataset.samples

        # 원본 이미지를 로드하기 위한 역정규화 transform
        inv_normalize = transforms.Normalize(
            mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225] if args.in_channels == 3 else [-0.5/0.5],
            std=[1/0.229, 1/0.224, 1/0.225] if args.in_channels == 3 else [1/0.5]
        )

        for batch_idx, (images, labels) in enumerate(test_loader):
            images = images.to(device)
            outputs = model(images)
            _, predicted_indices = torch.max(outputs.data, 1)

            if args.save_attention_maps:
                all_attns = []
                if hasattr(model, 'classifier') and hasattr(model.classifier, 'model') and hasattr(model.classifier.model, 'backbone') and hasattr(model.classifier.model.backbone, 'decoder'):
                    for layer in model.classifier.model.backbone.decoder.layers:
                        if hasattr(layer, 'attn'):
                            all_attns.append(layer.attn.cpu().numpy())
                
                if not all_attns: continue

                for sample_in_batch_idx in range(images.shape[0]):
                    global_sample_idx = batch_idx * args.batch_size + sample_in_batch_idx
                    original_idx = test_indices[global_sample_idx]
                    filepath = original_samples[original_idx][0]

                    # 원본 이미지 로드 및 역정규화
                    original_image_tensor = images[sample_in_batch_idx].cpu()
                    original_image_tensor = inv_normalize(original_image_tensor)
                    original_image = transforms.ToPILImage()(original_image_tensor).convert("RGB" if args.in_channels == 3 else "L")


                    actual_class = class_names[labels[sample_in_batch_idx]]
                    predicted_class = class_names[predicted_indices[sample_in_batch_idx]]

                    # GridSpec을 사용하여 원본 이미지와 어텐션 맵 레이아웃 구성
                    fig = plt.figure(figsize=(args.n_heads * 5, args.d_layers * 2.5 + 2))
                    gs = fig.add_gridspec(args.d_layers + 1, args.n_heads, height_ratios=[1.5] + [1]*args.d_layers, hspace=0.8, wspace=0.3)

                    # 원본 이미지 표시
                    ax_orig = fig.add_subplot(gs[0, 0])
                    ax_orig.imshow(original_image, cmap='gray' if args.in_channels == 1 else None)
                    ax_orig.set_title("Original Image", fontsize=12)
                    ax_orig.axis('off')
                    
                    title = f"File: {os.path.basename(filepath)} | Actual: {actual_class} | Predicted: {predicted_class}"
                    full_title = f"{title}\n{range_text}"
                    fig.suptitle(full_title, fontsize=16, y=1.02)

                    for layer_idx in range(args.d_layers):
                        for head_idx in range(args.n_heads):
                            ax = fig.add_subplot(gs[layer_idx + 1, head_idx])
                            attn_data = all_attns[layer_idx][sample_in_batch_idx, 0, head_idx, :]
                            
                            # 컬러바를 히트맵 아래에 별도로 생성
                            sns.heatmap(attn_data.reshape(1, -1), cmap='viridis', ax=ax, cbar=False)
                            
                            # 컬러바를 위한 새로운 축 생성 (히트맵 아래에 위치)
                            cax = ax.inset_axes([0, -0.5, 1, 0.1]) # [x, y, width, height]
                            fig.colorbar(ax.collections[0], cax=cax, orientation='horizontal')

                            ax.set_yticklabels([])
                            ax.set_xlabel('')
                            ax.set_ylabel('')

                            if layer_idx == 0: # 헤더 제목은 첫 줄에만 표시
                                ax.set_title(f'Head {head_idx+1}')
                            if head_idx == 0:
                                ax.set_ylabel(f'Layer {layer_idx+1}', rotation=90, size='large', labelpad=20)

                    save_path = os.path.join(attention_map_dir, f'sample_{global_sample_idx}.png')
                    plt.savefig(save_path, bbox_inches='tight')
                    plt.close(fig)

    logging.info("테스트셋에 대한 성능 평가를 시작합니다.")
    evaluate(model, test_loader, device)

# =============================================================================
# 4. 메인 실행 블록
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CATS 기반 이미지 분류기")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='실행 모드 (train 또는 inference)')
    parser.add_argument('--data_dir', type=str, default='./Refined_mix', help='데이터셋 폴더 경로')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='총 에포크 수')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='저장/로드할 모델 파일 이름')
    parser.add_argument('--in_channels', type=int, default=1, choices=[1, 3], help='입력 이미지 채널 수 (1: 흑백, 3: RGB)')
    parser.add_argument('--img_size', type=int, default=480, help='입력 이미지 크기')
    parser.add_argument('--patch_len', type=int, default=32, help='Patch length for CATS model')
    parser.add_argument('--emb_dim', type=int, default=24, help='Embedding dimension for CATS model')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads for CATS model')
    parser.add_argument('--d_layers', type=int, default=2, help='Number of decoder layers for CATS model')
    parser.add_argument('--patch_size', type=int, default=120, help='Patch size for image encoder')
    parser.add_argument('--save_attention_maps', action='store_true', help='추론 시 어텐션 맵을 저장할지 여부')
    
    cli_args = parser.parse_args()
    
    setup_logging(cli_args.data_dir)

    logging.info("===== 실행 인자 =====")
    for arg, value in vars(cli_args).items():
        logging.info(f"{arg}: {value}")
    logging.info("=====================")
    
    # --- 공통 파라미터 설정 ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = cli_args.img_size
    in_channels = cli_args.in_channels

    # --- 데이터 준비 (동적으로 변경) ---
    if in_channels == 1: 
        normalize = transforms.Normalize(mean=[0.5], std=[0.5])
    else: # 3일 때는 컬러 전용 정규화
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    train_transform_list = [
        transforms.Resize((int(img_size*1.1), int(img_size*1.1))),
        transforms.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05)),
        transforms.RandomResizedCrop(img_size, scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.RandomHorizontalFlip(p=0.5),
    ]
    if in_channels == 1:
        train_transform_list.append(transforms.Grayscale(num_output_channels=1))
    
    train_transform_list.extend([transforms.ToTensor(), normalize])
    train_transform = transforms.Compose(train_transform_list)


    test_transform_list = [
        transforms.Resize((img_size, img_size)),
    ]
    if in_channels == 1:
        test_transform_list.append(transforms.Grayscale(num_output_channels=1))

    test_transform_list.extend([transforms.ToTensor(), normalize])
    test_transform = transforms.Compose(test_transform_list)
    
    try:
        base_dataset = datasets.ImageFolder(root=cli_args.data_dir)
        targets = base_dataset.targets
        
        splitter = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
        train_idx, test_idx = next(splitter.split(range(len(targets)), targets))

        train_dataset = Subset(datasets.ImageFolder(root=cli_args.data_dir, transform=train_transform), train_idx)
        test_dataset = Subset(datasets.ImageFolder(root=cli_args.data_dir, transform=test_transform), test_idx)

        train_loader = DataLoader(train_dataset, batch_size=cli_args.batch_size, shuffle=True, num_workers=4)
        test_loader = DataLoader(test_dataset, batch_size=cli_args.batch_size, shuffle=False, num_workers=4)
        logging.info(f"데이터 로딩 완료. 훈련 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개")
    except FileNotFoundError:
        logging.error(f"데이터 폴더 '{cli_args.data_dir}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        exit()

    # --- 모델 구성 (동적으로 변경) ---
    num_labels = len(base_dataset.classes)
    
    patch_num = (cli_args.img_size // cli_args.patch_size) ** 2
    
    if in_channels == 3: 
        seq_len = (patch_num * cli_args.patch_len) * 3
        encoder_in_channels = 1 # 채널 독립성 모델은 1채널씩 처리하는 인코더 필요
    else:
        seq_len = patch_num * cli_args.patch_len
        encoder_in_channels = 1 # 채널 독립성 모델은 1채널씩 처리하는 인코더 필요

    cats_params = {
        'seq_len': seq_len, 'pred_len': num_labels, 'd_layers': cli_args.d_layers,
        'dec_in': 1, 'd_model': cli_args.emb_dim, 'd_ff': cli_args.emb_dim * 2, 'n_heads': cli_args.n_heads,
        'patch_len': cli_args.patch_len, 'stride': cli_args.patch_len, 'classification': True,
        'dropout': 0.1, 'query_independence': True, 'padding_patch': 'end',
        'store_attn': cli_args.save_attention_maps, 'QAM_end': 0.5, 'QAM_start': 0.0, 'features': 'S',
        'des': 'Exp', 'itr': 1,
    }
    # query_independence': True는 한 채널에 대한 쿼리를 다른 채널들에도 공유하는 옵션. 하지만 지금은 이미지를 1차원으로 펴서 CATS에 입력하기 때문에 true나 false가 의미없다.
    cats_args = SimpleNamespace(**cats_params)

    encoder = PatchConvEncoder(in_channels=encoder_in_channels, img_size=cli_args.img_size, patch_size=cli_args.patch_size, hidden_dim=cli_args.patch_len)
    classifier = CATS_Model(args=cats_args)
    
    if in_channels == 3:
        model = XModel_ChannelIndependent(encoder, classifier).to(device)
    else:
        model = XModel(encoder, classifier).to(device)
    
    # --- 모드에 따라 실행 ---
    if cli_args.mode == 'train':
        train(cli_args, model, train_loader, test_loader, device)
    elif cli_args.mode == 'inference':
        inference(cli_args, model, test_loader, device, base_dataset.classes)