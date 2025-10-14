python run.py --mode train --batch_size 4 --epochs 10

python run.py --mode inference



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="CATS 기반 이미지 분류기")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'inference'], help='실행 모드 (train 또는 inference)')
    parser.add_argument('--data_dir', type=str, default='./Refined_mix', help='데이터셋 폴더 경로')
    parser.add_argument('--batch_size', type=int, default=8, help='배치 크기')
    parser.add_argument('--epochs', type=int, default=100, help='총 에포크 수')
    parser.add_argument('--lr', type=float, default=0.001, help='학습률')
    parser.add_argument('--model_path', type=str, default='best_model.pth', help='저장/로드할 모델 파일 이름')
    parser.add_argument('--in_channels', type=int, default=3, choices=[1, 3], help='입력 이미지 채널 수 (1: 흑백, 3: RGB)')
    parser.add_argument('--img_size', type=int, default=480, help='입력 이미지 크기')
    parser.add_argument('--patch_len', type=int, default=32, help='Patch length for CATS model')
    parser.add_argument('--emb_dim', type=int, default=24, help='Embedding dimension for CATS model')
    parser.add_argument('--n_heads', type=int, default=4, help='Number of heads for CATS model')
    parser.add_argument('--d_layers', type=int, default=2, help='Number of decoder layers for CATS model')
    parser.add_argument('--patch_size', type=int, default=120, help='Patch size for image encoder')
    parser.add_argument('--save_attention_maps', action='store_true', help='추론 시 어텐션 맵을 저장할지 여부')
    

python run_ci.py --mode train --data_dir "./tap_new" --epochs 20 --in_channels 3
python run_ci.py --mode train --data_dir "./Refined_mix" --epochs 20 --in_channels 3

python run_ci.py --mode inference --data_dir ./tap_new --save_attention_maps
python run_ci.py --mode inference --data_dir ./Refined_mix --save_attention_maps

python run_ci.py --mode inference --data_dir ./Refined_mix

---

python run_ci.py --mode inference --data_dir ./tap_new --n_heads 3 --save_attention_maps

python run_ci.py --mode inference --data_dir ./Refined_mix --n_heads 3 --save_attention_maps

python run_ci.py --mode train --data_dir ./Refined_mix --n_heads 3 --epoch 2
