import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# GEGLU (Gated Enhanced Gated Linear Unit) 액티베이션 함수를 구현한 클래스입니다.
# 일반적인 ReLU나 GELU와 달리, 입력의 일부를 게이트로 사용하여 동적으로 출력을 조절하는 특징이 있습니다.
class GEGLU(nn.Module):
    # 이 클래스의 순전파 로직을 정의합니다. 입력 텐서를 받아 GEGLU 연산을 수행합니다.
    def forward(self, x):
        x, gate = x.chunk(2, dim=-1)
        # 입력 텐서 `x`의 마지막 차원(`dim=-1`)을 기준으로 x를 두 개의 동일한 크기의 청크(chunk)로 나눕니다.
        # `x`는 주 데이터 경로가 되고, `gate`는 게이트 역할을 합니다. 수학적으로 x_in = [x, gate] 입니다.
        return x * F.gelu(gate)
        # 주 데이터 경로 `x`와, `gate`에 GELU(Gaussian Error Linear Unit) 활성화 함수를 적용한 결과를 요소별로 곱합니다.
        # 이 게이팅 메커니즘은 gate가 x에서 어떤 요소를 증폭하거나 줄일지 스스로 판단합니다.
        # 덕분에 모델이 더 복잡한 패턴을 학습하도록 돕습니다. 수학식은 Output = x * GELU(gate) 입니다.

# Query-Adaptive Masking (QAM)을 구현한 클래스입니다.
# 학습 중에 입력 텐서의 일부를 동적으로 마스킹(0으로 만듦)하여 레귤러라이제이션(regularization) 효과를 줍니다.
# 특히, 마스킹 확률이 차원을 따라 선형적으로 변하는 특징이 있습니다.
class QueryAdaptiveMasking(nn.Module):
    # QAM 클래스의 생성자입니다. 마스킹을 적용할 차원과 확률 범위를 설정합니다.
    # 입력 텐서 x: (배치, N_T, d_model)
    def __init__(self, dim=1, start_prob =0.1, end_prob =0.5):
        super().__init__()
        # `nn.Module`의 생성자를 호출하여 PyTorch 모델로서의 기본 기능을 초기화합니다.
        self.dim = dim
        # 마스킹 목표 차원을 저장합니다. `dim=1`은 텐서의 두 번째 차원 N_T를 의미합니다.
        self.start_prob = start_prob
        # 마스킹 확률의 시작 값을 저장합니다. 이 값은 지정된 차원의 첫 번째 요소에 적용됩니다.
        self.end_prob = end_prob
        # 마스킹 확률의 끝 값을 저장합니다. 이 값은 지정된 차원의 마지막 요소에 적용됩니다.
    # QAM의 순전파 로직을 정의합니다.
    def forward(self, x): # 입력 텐서 x: (배치, N_T, d_model)
        if not self.training:
            # 모델이 평가 모드(`model.eval()`)일 때는 마스킹을 적용하지 않습니다.
            return x
            # 입력을 그대로 반환하여 예측 시에는 일관된 결과를 얻도록 합니다.
        # 모델이 학습 모드(`model.train()`)일 때만 마스킹을 적용합니다.
        else:
            size = x.shape[self.dim]
            # 마스킹을 적용할 차원의 크기, 즉 미래 예측 패치의 개수(N_T)를 가져옵니다.
            dropout_prob = torch.linspace(self.start_prob,self.end_prob,steps=size,device=x.device).view([-1 if i == self.dim else 1 for i in range(x.dim())])
            # `torch.linspace`를 사용하여 `start_prob`에서 `end_prob`까지 `size` 개수만큼 선형적으로 증가하는 드롭아웃 확률 시퀀스를 생성합니다.
            # 각 패치마다 선형적으로 다른 드롭아웃 확률을 적용시킨다.
            # i번째 요소의 드롭아웃 확률은 p_i = p_start + (p_end - p_start) * (i / (size - 1)) 입니다.
            # `.view`를 통해 이 확률 텐서 x의 모양을 입력 텐서 `x`와 브로드캐스팅이 가능하도록 조정합니다.
            mask = torch.bernoulli(1 - dropout_prob).expand_as(x)
            # `1 - dropout_prob` 확률(p)에 따라 1(성공) 또는 0(실패)의 값을 갖는 베르누이 분포로부터 마스크를 생성합니다. 즉, 각 요소는 1-p의 확률로 1이 되고 p의 확률로 0이 됩니다.
            # 이 마스크는 입력 `x`와 동일한 크기로 확장됩니다.
            return x*mask
            # 생성된 마스크를 입력 텐서 `x`에 요소별로 곱하여 특정 요소들을 0으로 만듭니다(마스킹). 수학식은 x_out = x * mask 입니다.

# 시계열 예측 모델의 핵심 구조인 백본(backbone)을 정의하는 클래스입니다.
# 입력 데이터를 패치로 나누고, 트랜스포머 기반의 인코더-디코더 구조를 통해 예측을 수행합니다.
class Model_backbone(nn.Module):
    # 모델 백본의 생성자입니다. 모델의 구조와 하이퍼파라미터를 초기화합니다.
    def __init__(self, num_encoder_patches:int, num_labels:int, featured_patch_dim:int=24, n_layers:int=3, emb_dim=128, n_heads=16, d_ff:int=256,
                 attn_dropout:float=0., dropout:float=0., res_attention:bool=True, store_attn:bool=False, QAM_start:float = 0.1,
                 QAM_end:float =0.5, **kwargs):
        
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        self.featured_patch_dim = featured_patch_dim
        
        num_decoder_patches = (num_labels+self.featured_patch_dim-1)//self.featured_patch_dim
        # 디코더에서 사용할 학습 가능한 쿼리 패치의 수를 계산합니다. `num_labels`를 `featured_patch_dim`으로 올림 나눗셈하여 구합니다.
        
        # --- 백본 모델(임베딩 및 디코더) 초기화 --- 
        self.backbone = Learnable_Query_Embedding(num_encoder_patches=num_encoder_patches, featured_patch_dim=self.featured_patch_dim, num_decoder_patches=num_decoder_patches,
                                n_layers=n_layers, emb_dim=emb_dim, n_heads=n_heads, d_ff=d_ff, positional_encoding=kwargs.get('positional_encoding', True),
                                attn_dropout=attn_dropout, dropout=dropout, QAM_start=QAM_start, QAM_end=QAM_end,
                                res_attention=res_attention, store_attn=store_attn, **kwargs)
        # `Learnable_Query_Embedding` 클래스를 사용하여 실제 트랜스포머 연산을 수행할 백본을 생성합니다.

        self.num_labels = num_labels
        # 모델이 예측해야 할 클래스의 수를 저장합니다.
        self.proj = Projection(emb_dim, self.featured_patch_dim)
        # 백본의 출력(`emb_dim` 차원)을 최종 예측 패치(`featured_patch_dim` 차원)로 변환하는 프로젝션 레이어를 생성합니다.
    
    # 모델 백본의 순전파 로직을 정의합니다.
    def forward(self, z): # 입력 z의 형태: [배치 크기, 입력 다변량 시계열 데이터의 변수 개수, 시퀀스 길이L]
        # 입력 z의 형태: [배치 크기, 인코더 패치 수, 특징 차원]
        
        # --- 모델 순전파 ---
        z = self.backbone(z)
        # 패치화된 입력을 백본 모델(`Learnable_Query_Embedding`)에 통과시킵니다.
        z = self.proj(z)
        # 백본의 출력을 프로젝션 레이어에 통과시켜 최종 예측을 생성합니다.
        
        return z
        # 최종 예측 결과를 반환합니다.
    
# 입력 패치와 학습 가능한 쿼리(learnable queries)를 임베딩하고 트랜스포머 디코더를 통해 예측을 수행하는 클래스입니다.
# 디코더에 입력할 seq_encoder_patches와 seq_decoder_patches를 생성합니다. 이후 CATS 모델의 순전파에서 이들이 디코더에 입력되어 예측값을 생성하고, 오차 계산 및 역전파를 통해 훈련됩니다.
# 이 파라미터는 처음에는 무작위 값으로 시작하지만, 훈련 과정을 통해 각 미래 패치를 예측하기 위한 유의미한 질문(쿼리)으로 학습되기 때문에 "학습 가능한 쿼리"라고 부릅니다.
class Learnable_Query_Embedding(nn.Module): 
    # 클래스의 생성자입니다.
    def __init__(self, num_encoder_patches, featured_patch_dim, num_decoder_patches, n_layers=3, emb_dim=128, n_heads=16, QAM_start=0.1, QAM_end=0.5, d_ff=256, 
                 attn_dropout=0., dropout=0., store_attn=False, res_attention=True, positional_encoding=True, **kwargs):
             
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        # --- 입력 인코딩 ---
        self.W_P = nn.Linear(featured_patch_dim, emb_dim)      
        # 입력 패치(`featured_patch_dim` 차원)를 모델의 은닉 상태 차원(`emb_dim`)으로 변환하는 선형 레이어(가중치 `W_P`)를 정의합니다.
        self.dropout = nn.Dropout(dropout)
        # 일반적인 드롭아웃 레이어를 정의합니다.

        # --- 학습 가능한 쿼리(Learnable Queries) 설정 ---
        self.learnable_queries = nn.Parameter(0.5*torch.randn(num_decoder_patches, featured_patch_dim))
        # 모든 변수가 공유하는 하나의 쿼리 셋을 학습 가능한 파라미터(nn.Parameter)로 생성합니다.
        # 더 좁은 범위에 집중시키 위해서 0.5를 곱하여 N(0,0.5^2)를 따르게 만든다.
        # 크기는 [디코더 패치 수, featured_patch_dim]입니다.
        
        # --- 학습 가능한 위치 인코딩 ---
        # 입력 시퀀스의 위치 정보를 제공하기 위해, '학습 가능한 위치 인코딩(Positional Encoding)'을 파라미터로 생성합니다.
        self.use_positional_encoding = positional_encoding
        if self.use_positional_encoding:
            # 사인/코사인 함수 대신 직접 학습하는 방식입니다.
            self.PE = nn.Parameter(0.04*torch.rand(num_encoder_patches, emb_dim)-0.02)
            # 0과 1 사이의 균등 분포(torch.rand)를 따르는 무작위 숫자를 요소로 [패치 개수, 임베딩 차원] 모양의 행렬을 생성합니다.
        else:
            self.PE = None
        # --- 디코더 ---
        self.decoder = Decoder(num_encoder_patches, emb_dim, n_heads, num_decoder_patches, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                               QAM_start=QAM_start, QAM_end=QAM_end, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        # 실제 어텐션 연산을 수행할 트랜스포머 디코더를 초기화합니다.
        
    # 순전파 로직을 정의합니다.
    def forward(self, x) -> Tensor:
        # 입력 x의 형태: [배치 크기, 인코더 패치 수, featured_patch_dim]
        bs = x.shape[0]

        # --- 1. 디코더에 입력할 입력 시퀀스 준비 (Key, Value) ---
        x = self.W_P(x)
        if self.use_positional_encoding:
            x = x + self.PE
        # x shape: [B, num_encoder_patches, emb_dim]

        seq_encoder_patches = self.dropout(x)
        # 인코딩된 입력 패치에 드롭아웃을 적용합니다.
         
        # --- 2. 디코더에 입력할 학습 가능한 쿼리 준비 (Query) ---
        learnable_queries = self.W_P(self.learnable_queries)
        # 학습 가능한 쿼리 `learnable_queries` 또한 동일한 선형 레이어 `W_P`로 임베딩합니다.
        
        seq_decoder_patches = learnable_queries.unsqueeze(0).repeat(bs, 1, 1)           
        # 공유된 쿼리를 배치 내 모든 샘플에 대해 동일하게 복제합니다.
        # learnable_queries: [num_decoder_patches, emb_dim]
        # -> [1, num_decoder_patches, emb_dim]
        # -> [bs, num_decoder_patches, emb_dim]
        
        # --- 3. 디코더 순전파 ---
        z = self.decoder(seq_encoder_patches, seq_decoder_patches)
        # 준비된 인코더 패치와 디코더 쿼리 패치를 디코더에 전달합니다.
        # z shape: [bs, num_decoder_patches, emb_dim]

        return z
            
# 트랜스포머 백본의 출력을 최종 예측 시계열 형태로 변환하는 프로젝션 클래스입니다.
class Projection(nn.Module):
    # 프로젝션 클래스의 생성자입니다.
    def __init__(self, emb_dim, featured_patch_dim):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        self.linear = nn.Linear(emb_dim, featured_patch_dim)
        # 트랜스포머의 은닉 상태 차원, 즉 임베딩 차원(`emb_dim`)을 `featured_patch_dim`으로 변환하는 선형 레이어를 정의합니다.
        self.flatten = nn.Flatten(start_dim=-2)
        # 예측 패치들을 하나의 연속된 시계열로 펼치기 위한 Flatten 레이어를 정의합니다.
    # 프로젝션의 순전파 로직을 정의합니다.
    def forward(self, x):
        # 입력 x의 형태: [B, num_decoder_patches, emb_dim]
        x = self.linear(x)
        # 입력 `x`를 선형 레이어에 통과시켜 차원을 변환합니다.
        # 결과: [B, num_decoder_patches, featured_patch_dim]
        
        # flatten을 적용하여 마지막 두 차원을 하나로 합칩니다.
        # [B, num_decoder_patches, D] -> [B, num_decoder_patches * D]
        x = self.flatten(x)
        return x.unsqueeze(1) # Classifier와 호환되도록 [B, 1, L*D] 형태로 만듭니다.
            
# 여러 개의 디코더 레이어로 구성된 트랜스포머 디코더 클래스입니다.
class Decoder(nn.Module):
    # 디코더의 생성자입니다.
    def __init__(self, num_encoder_patches, emb_dim, n_heads, num_decoder_patches, d_ff=None, attn_dropout=0., dropout=0., QAM_start = 0.1, QAM_end =0.5,
                        res_attention=False, n_layers=1, store_attn=False):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        self.layers = nn.ModuleList([DecoderLayer(num_encoder_patches, emb_dim, num_decoder_patches, n_heads=n_heads, d_ff=d_ff, QAM_start=QAM_start, QAM_end=QAM_end,
                                                      attn_dropout=attn_dropout, dropout=dropout, res_attention=res_attention,
                                                      store_attn=store_attn) for i in range(n_layers)])
        # `n_layers` 개수만큼의 `DecoderLayer`를 `nn.ModuleList`로 묶어 관리합니다.
        
        self.res_attention = res_attention
        # 잔차 어텐션(residual attention) 메커니즘 사용 여부를 저장합니다. (어텐션 스코어를 다음 레이어에 더해주는 기법)
    # 디코더의 순전파 로직을 정의합니다.
    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor):
        scores = None
        # 잔차 어텐션에서 이전 레이어의 어텐션 스코어를 전달하기 위한 변수를 초기화합니다.
        
        if self.res_attention:
            # 잔차 어텐션을 사용하는 경우,
            for mod in self.layers: _, seq_decoder, scores = mod(seq_encoder, seq_decoder, prev=scores)
            return seq_decoder
        
        else:
            # 잔차 어텐션을 사용하지 않는 경우,
            for mod in self.layers: _, seq_decoder = mod(seq_encoder, seq_decoder)
            return seq_decoder

# 트랜스포머 디코더의 단일 레이어를 정의하는 클래스입니다.
# 크로스-어텐션(Cross-Attention)과 피드포워드 네트워크(Feed-Forward Network)로 구성됩니다.
class DecoderLayer(nn.Module):
    # 디코더 레이어의 생성자입니다.
    def __init__(self, num_encoder_patches, emb_dim, num_decoder_patches, n_heads, d_ff=256, store_attn=False, QAM_start = 0.1, QAM_end =0.5,
                 attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        assert not emb_dim%n_heads, f"emb_dim ({emb_dim}) must be divisible by n_heads ({n_heads})"
        # `emb_dim`이 `n_heads`로 나누어 떨어져야 멀티헤드 어텐션이 가능하므로, 이를 확인합니다.
        
        # --- 크로스-어텐션 블록 ---
        self.res_attention = res_attention
        # 잔차 어텐션 사용 여부를 저장합니다.
        self.cross_attn = _MultiheadAttention(emb_dim, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        # 멀티헤드 크로스-어텐션 모듈을 초기화한다.
        self.dropout_attn = QueryAdaptiveMasking(dim=1, start_prob=QAM_start, end_prob=QAM_end)
        # 어텐션 출력에 적용할 Query-Adaptive Masking 레이어를 정의합니다.
        self.norm_attn = nn.LayerNorm(emb_dim)
        # 어텐션 블록의 잔차 연결(add) 후 적용될 레이어 정규화(Layer Normalization)를 정의합니다.

        # --- 피드포워드 네트워크 블록 ---
        # 위치별 피드포워드 네트워크(FFN)를 `nn.Sequential`로 정의합니다.
        self.ffn = nn.Sequential(nn.Linear(emb_dim, d_ff, bias=bias), # 1. emb_dim -> d_ff 확장
                                GEGLU(),                             # 2. GEGLU 활성화 함수 (이 활성함수를 거친 직후 차원이 절반이 됨))
                                nn.Dropout(dropout),                 # 3. 드롭아웃
                                nn.Linear(d_ff//2, emb_dim, bias=bias)) # 4. d_ff/2 -> emb_dim 축소 
        self.dropout_ffn = QueryAdaptiveMasking(dim=1, start_prob=QAM_start, end_prob=QAM_end)
        # FFN 출력에 적용할 Query-Adaptive Masking 레이어를 정의합니다.
        self.norm_ffn = nn.LayerNorm(emb_dim)
        # FFN 블록의 잔차 연결(add) 후 적용될 레이어 정규화를 정의합니다.
        
        self.store_attn = store_attn
        # 어텐션 가중치를 시각화 등의 목적으로 저장할지 여부를 결정합니다.

    # 디코더 레이어의 순전파 로직을 정의합니다.
    def forward(self, seq_encoder:Tensor, seq_decoder:Tensor, prev=None) -> Tensor:
        # `seq_decoder`는 쿼리(Q), `seq_encoder`는 키(K)와 값(V)으로 사용됩니다.
        
        # --- 멀티헤드 크로스-어텐션 ---
        if self.res_attention:
            # 잔차 어텐션을 사용하는 경우,
            decoder_out, attn, scores = self.cross_attn(seq_decoder, seq_encoder, seq_encoder, prev)
            # 어텐션 모듈은 (출력, 어텐션 가중치, 어텐션 스코어)를 반환합니다.
        else:
            # 잔차 어텐션을 사용하지 않는 경우,
            decoder_out, attn = self.cross_attn(seq_decoder, seq_encoder, seq_encoder)
            # 어텐션 모듈은 (출력, 어텐션 가중치)를 반환합니다.
        if self.store_attn:
            # 어텐션 가중치를 저장하도록 설정된 경우,
            self.attn = attn
            # 계산된 어텐션 가중치를 `self.attn`에 저장합니다.
        
        # --- 첫 번째 Add & Norm ---
        seq_decoder = seq_decoder + self.dropout_attn(decoder_out)
        # 1번째 Residual Connection: 어텐션 출력에 드롭아웃을 적용하고, 이를 입력에 더합니다.
        seq_decoder = self.norm_attn(seq_decoder)
        # 레이어 정규화를 적용합니다.
        
        # --- 피드포워드 네트워크 ---
        ffn_out = self.ffn(seq_decoder)
        # 정규화된 결과를 FFN에 통과시킵니다.

        # --- 두 번째 Add & Norm ---
        seq_decoder = seq_decoder + self.dropout_ffn(ffn_out)  
        # 2번째 Residual Connection: FFN의 출력에 드롭아웃을 적용하고, 이를 FFN의 입력에 더합니다.
        seq_decoder = self.norm_ffn(seq_decoder)
        # 레이어 정규화를 적용합니다.
        
        if self.res_attention: return seq_encoder, seq_decoder, scores
        else: return seq_encoder, seq_decoder

# 멀티헤드 어텐션 메커니즘을 구현한 내부 클래스입니다.
class _MultiheadAttention(nn.Module):
    # 멀티헤드 어텐션의 생성자입니다.
    def __init__(self, emb_dim, n_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True):
        """
        멀티헤드 어텐션 레이어
        입력 형태:
            Q: [배치 크기, 디코더 패치 수, emb_dim]
            K, V: [배치 크기, 인코더 패치 수, emb_dim]
        """
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        d_h = emb_dim // n_heads
        # 각 어텐션 헤드의 차원을 계산합니다.
        self.scale = d_h**-0.5
        # 어텐션 스코어를 스케일링하기 위한 팩터입니다. d_k의 제곱근의 역수(1/sqrt(d_k))를 사용합니다.
        # 이는 Q, K 내적 값의 분산이 d_k에 비례하여 커지는 것을 방지하여, softmax 함수의 기울기 소실(gradient vanishing) 문제를 완화합니다.
        self.n_heads, self.d_h = n_heads, d_h
        # 헤드의 수와 각 헤드의 차원을 저장합니다.

        # 입력 Q, K, V를 `emb_dim` 차원에서 `n_heads * d_h` 차원으로 변환하는 선형 레이어들을 정의합니다.
        self.W_Q = nn.Linear(emb_dim, d_h * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(emb_dim, d_h * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(emb_dim, d_h * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        # 잔차 어텐션 사용 여부를 저장합니다.
        self.attn_dropout = nn.Dropout(attn_dropout)
        # 계산된 어텐션 가중치에 적용될 드롭아웃 레이어를 정의합니다.
        
        # n_heads * d_h 차원에서 emb_dim 차원으로 변환시킨 후, 드롭아웃을 적용하는 출력 레이어를 정의합니다.
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_h, emb_dim), nn.Dropout(proj_dropout))

    # 멀티헤드 어텐션의 순전파 로직을 정의합니다.
    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):
        bs = Q.size(0)
        # 입력 쿼리(Q)에서 배치 크기를 가져옵니다.
        
        # --- Q, K, V 선형 변환 및 헤드 분할 ---
        # 
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_h)     
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_h) 
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_h) 
        # 선형 변환 후 `view`를 통해 멀티 헤드로 나눕니다.
        # 결과: [배치 크기, 패치 수, 멀티 헤드 개수, 헤드 차원]

        # --- 어텐션 스코어 계산 ---
        attn_scores = torch.einsum('bphd, bshd -> bphs', q_s, k_s) * self.scale
        # `torch.einsum` (아인슈타인 표기법)을 사용하여 Q와 K의 내적을 효율적으로 계산합니다.
        # b: batch, p: Q의 시퀀스 차원, s: K의 시퀀스 차원, h: head, d: dimension
        # 이는 행렬 곱 Q * K^T 와 동일한 연산입니다.
        # 계산된 스코어를 `scale` 팩터로 스케일링합니다. Attention(Q, K) = (Q * K^T) / sqrt(d_k)
        
        if prev is not None: attn_scores = attn_scores + prev
        # 만약 이전 레이어의 어텐션 스코어(`prev`)가 주어지면, 현재 스코어에 더합니다 (잔차 어텐션).
        
        # --- 어텐션 가중치 및 출력 계산 ---
        attn_weights = F.softmax(attn_scores, dim=-1)
        # 어텐션 스코어에 소프트맥스 함수를 적용하여 확률 분포 형태의 어텐션 가중치를 얻습니다. 각 쿼리에 대한 키의 중요도를 나타냅니다.
        attn_weights = self.attn_dropout(attn_weights)
        # 어텐션 가중치에 드롭아웃을 적용합니다.

        output = torch.einsum('bphs, bshd -> bphd', attn_weights, v_s)
        # 어텐션 가중치와 값(V)을 `einsum`으로 곱하여 최종 출력을 계산합니다.
        # 이는 Attention Weight * V 와 동일한 연산입니다.
        output = output.contiguous().view(bs, -1, self.n_heads*self.d_h)
        # 나뉘었던 헤드들을 `contiguous()`와 `view`를 통해 다시 하나의 텐서로 합칩니다.
        
        output = self.to_out(output)
        # 최종 출력 레이어를 통과시킵니다.

        if self.res_attention: return output, attn_weights, attn_scores
        else: return output, attn_weights

# 전체 모델을 구성하고 순전파를 정의하는 메인 클래스입니다.
# 하이퍼파라미터를 인자로 받아 `Model_backbone`을 초기화하고 데이터의 입출력 형식을 관리합니다.
class Model(nn.Module):
    # 전체 모델의 생성자입니다.
    def __init__(self, args, **kwargs):
        
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        # --- 하이퍼파라미터 로드 ---
        # `args` 객체로부터 모델 구성에 필요한 모든 하이퍼파라미터를 가져옵니다.
        num_encoder_patches = args.num_encoder_patches # 인코더 패치의 수
        num_labels = args.num_labels # 예측할 클래스의 수
        n_layers = args.d_layers         # 트랜스포머 디코더의 레이어 수
        n_heads = args.n_heads           # 멀티헤드 어텐션의 헤드 수
        emb_dim = args.emb_dim           # 모델의 은닉 상태 차원
        d_ff = args.d_ff                 # 피드포워드 네트워크의 내부 차원
        dropout = args.dropout           # 드롭아웃 비율
        positional_encoding = args.positional_encoding # 위치 인코딩 사용 여부
        store_attn = args.store_attn     # 어텐션 가중치 저장 여부
        QAM_start = getattr(args, 'QAM_start', 0.0) # QAM 시작 확률
        QAM_end = getattr(args, 'QAM_end', 0.0)     # QAM 끝 확률

        # 로드한 하이퍼파라미터들을 사용하여 `Model_backbone`을 초기화합니다. 
        self.model = Model_backbone(num_encoder_patches=num_encoder_patches, num_labels=num_labels, featured_patch_dim=args.featured_patch_dim, n_layers=n_layers,
                                    emb_dim=emb_dim, n_heads=n_heads, d_ff=d_ff, dropout=dropout, positional_encoding=positional_encoding, store_attn=store_attn, QAM_start=QAM_start, QAM_end=QAM_end, **kwargs)
        
    
    # 전체 모델의 순전파 로직을 정의합니다.
    def forward(self, x): # 입력 x의 형태: [배치 크기, 인코더 패치 수, 특징 차원]

        # 백본 모델에 입력을 통과시켜 특징 텐서를 추출합니다.
        features = self.model(x)
        # 결과 features의 형태: [B, 1, num_decoder_patches * featured_patch_dim]
        return features