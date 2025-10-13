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
    def __init__(self, c_in:int, seq_len:int, pred_len:int, patch_len:int=24, stride:int=24, n_layers:int=3, d_model=128, n_heads=16, d_ff:int=256, 
                 attn_dropout:float=0., dropout:float=0., res_attention:bool=True, independence:bool=False, store_attn:bool=False, QAM_start:float = 0.1, 
                 QAM_end:float =0.5, padding_patch = None, **kwargs):
        
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        # --- 패치화(Patching) 관련 설정 ---
        self.patch_len = patch_len
        # 입력 시계열을 나눌 각 패치의 길이를 저장합니다.
        self.stride = stride
        # 패치를 추출할 때의 보폭(stride)을 저장합니다.
        self.padding_patch = padding_patch
        # 패딩 전략을 저장합니다. `None` 또는 'end(시계열 데이터 맨 뒤에 추가)'가 될 수 있습니다.
        
        pred_patch_num = (pred_len+patch_len-1)//patch_len
        # 미래 예측 패치의 개수를 계산합니다. `pred_len`을 `patch_len`으로 올림 나눗셈(ceiling division)하여 구합니다.
        # (pred_len + patch_len - 1) // patch_len은 math.ceil(pred_len / patch_len)과 동일한 결과를 정수 연산으로 수행합니다.
        seq_patch_num = int((seq_len - patch_len)/stride + 1)
        # 입력 시계열에서 생성될 과거 패치의 개수를 계산합니다. 컨볼루션 출력 크기 계산과 유사합니다.
        # N_out = floor((N_in - kernel_size) / stride + 1) 공식에 해당합니다.
        
        if padding_patch == 'end':
            # 만약 'end' 패딩 전략이 사용된다면,
            self.padding_patch_layer = nn.ReplicationPad1d((0, stride)) 
            # 시계열의 끝부분에 `stride`만큼의 값을 복제하여 패딩하는 레이어를 추가합니다. 이는 시퀀스 길이를 유지하는 데 도움이 됩니다.
            seq_patch_num += 1
            # 패딩으로 인해 시퀀스 패치의 수가 하나 증가합니다.
        
        # --- 백본 모델 초기화 ---
        self.backbone = Dummy_Embedding(c_in, seq_patch_num=seq_patch_num, patch_len=patch_len, pred_patch_num=pred_patch_num,
                                n_layers=n_layers, d_model=d_model, n_heads=n_heads, d_ff=d_ff,
                                attn_dropout=attn_dropout, dropout=dropout, QAM_start=QAM_start, QAM_end=QAM_end,
                                res_attention=res_attention, independence = independence, store_attn=store_attn, **kwargs)
        # `Dummy_Embedding` 클래스를 사용하여 실제 트랜스포머 연산을 수행할 백본을 생성합니다.

        self.n_vars = c_in
        # 입력 시계열의 변수(채널) 개수를 저장합니다.
        self.pred_len = pred_len
        # 모델이 예측해야 할 타임스텝의 길이를 저장합니다.
        self.proj = Projection(d_model, patch_len)
        # 백본의 출력(`d_model` 차원)을 최종 예측 패치(`patch_len` 차원)로 변환하는 프로젝션 레이어를 생성합니다.
    
    # 모델 백본의 순전파 로직을 정의합니다.
    def forward(self, z): # 입력 z의 형태: [배치 크기, 입력 다변량 시계열 데이터의 변수 개수, 시퀀스 길이L]
        # --- 인스턴스 정규화(Instance Normalization) ---
        # 각 시계열 인스턴스(샘플)별로 독립적으로 정규화를 수행하여 분포 차이로 인한 학습 불안정성을 줄입니다.
        mean = z.mean(2, keepdim=True)
        # 시퀀스 길이 차원(dim=2)에 대해 평균(μ)을 계산합니다. `keepdim=True`는 차원을 유지합니다.
        std = torch.sqrt(torch.var(z, dim=2, keepdim=True, unbiased=False) + 1e-5)
        # 시퀀스 길이 차원에 대해 분산(σ^2)을 계산하고, 0으로 나누는 것을 방지하기 위해 작은 값(epsilon, 1e-5)을 더한 후 제곱근을 취해 표준편차(σ)를 구합니다.
        z = (z - mean)/std 
        # 입력 데이터에서 평균을 빼고 표준편차로 나누어 표준 정규분포에 가깝게 정규화를 수행합니다. z_norm = (z - μ) / σ.
            
        # --- 패치화(Patching) ---
        if self.padding_patch == 'end':
            # 'end' 패딩이 설정된 경우,
            z = self.padding_patch_layer(z)
            # 정의된 패딩 레이어를 입력에 적용합니다.
        z = z.unfold(dimension=-1, size=self.patch_len, step=self.stride)
        # `unfold` 함수를 사용하여 시계열을 지정된 `patch_len`과 `stride`에 따라 패치들로 나눕니다.
        # 결과 z의 형태: [배치 크기, 변수 수, 시퀀스 패치 수, 패치 길이]

        # --- 모델 순전파 ---
        z = self.backbone(z)
        # 패치화된 입력을 백본 모델(`Dummy_Embedding`)에 통과시킵니다.
        # 결과 z의 형태: [배치 크기, 변수 수, 예측 패치 수, d_model]
        z = self.proj(z)
        # 백본의 출력을 프로젝션 레이어에 통과시켜 최종 예측을 생성합니다.
        # 결과 z의 형태: [배치 크기, 변수 수, 예측 길이]
        
        # --- 역정규화(Denormalization) ---
        # 모델의 출력을 원래 데이터의 스케일로 되돌리는 과정입니다.
        z = z * (std[:, :, 0].unsqueeze(2).repeat(1, 1, self.pred_len))
        # 정규화 시 사용했던 표준편차(σ)를 다시 곱합니다. z_denorm = z_norm * σ.
        # `unsqueeze`와 `repeat`은 텐서의 차원을 예측 길이에 맞게 브로드캐스팅하기 위해 사용됩니다.
        z = z + (mean[:, :, 0].unsqueeze(2).repeat(1, 1, self.pred_len))
        # 정규화 시 뺐던 평균(μ)을 다시 더합니다. z_final = z_denorm + μ.
        
        return z
        # 최종 예측 결과를 반환합니다.
    
# 입력 패치와 dummy(미래 패치를 예측하기 위해 필요한 학습 가능한 쿼리 파라미터)를 임베딩하고 트랜스포머 디코더를 통해 예측을 수행하는 클래스입니다.
# 디코더에 입력할 seq_patch와 pred_patch를 생성한다. 이후 CATS 모델의 순전파에서 seq_patch와 pred_patch가 디코더에 입력되며 예측값을 생성한다. 그리고 오차를 계산하고 역전파로 훈련한다.
# pred_patch의 시작값 이름이 'dummy'인 이유는 이 파라미터가 처음에는 아무 의미 없는 무작위 값(더미 데이터)으로 시작하기 때문이다. 물론 위 훈련 과정을 통해 각 미래 패치를 예측하기 위한 유의미한 질문(쿼리)으로 학습되기 때문입니다.
class Dummy_Embedding(nn.Module): 
    # 클래스의 생성자입니다.
    def __init__(self, c_in, seq_patch_num, patch_len, pred_patch_num, n_layers=3, d_model=128, n_heads=16, QAM_start = 0.1, QAM_end =0.5,
                 d_ff=256, attn_dropout=0., dropout=0., store_attn=False, res_attention=True, independence = False, **kwargs):
             
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        # --- 입력 인코딩 ---
        self.W_P = nn.Linear(patch_len, d_model)      
        # 입력 패치(`patch_len` 차원)를 모델의 은닉 상태 차원(`d_model`)으로 변환하는 선형 레이어(가중치 `W_P`)를 정의합니다.
        self.dropout = nn.Dropout(dropout)
        # 일반적인 드롭아웃 레이어를 정의합니다.

        # --- 더미 입력(학습 가능한 쿼리) 설정 ---
        self.independence = independence
        # 변수(채널)들이 서로 독립적으로 처리될지 여부를 저장합니다.
        if self.independence:
            # 만약 변수들이 독립적이라면,
            self.dummies = nn.Parameter(0.5*torch.randn(pred_patch_num, patch_len))
            # 모든 변수가 공유하는 하나의 더미 입력 셋을 학습 가능한 파라미터(nn.Parameter)로 생성합니다.
            # 더 좁은 범위에 집중시키 위해서 0.5를 곱하여 N(0,0.5^2)를 따르게 만든다.
            # -> 초기 가중치 값이 0근처에 더 집중되게 만듦으로써 초기 가중치가 커서 발생할 문제 방지 (그래디언트 폭주/소실. 활성함수 포화)
            # 크기는 [예측 패치 수, 패치 길이]입니다.
        else:
            # 변수들이 서로 종속적이라면,
            self.dummies = nn.Parameter(0.5*torch.randn(c_in, pred_patch_num, patch_len))
            # 각 변수별로 별도의 더미 입력 셋을 학습 가능한 파라미터(nn.Parameter)로 생성합니다.
            # 크기는 [변수 수, 예측 패치 수, 패치 길이]입니다.
        
        # --- 학습 가능한 위치 인코딩 ---
        # 입력 시퀀스의 위치 정보를 제공하기 위해, '학습 가능한 위치 인코딩(Positional Encoding)'을 파라미터로 생성합니다.
        # 사인/코사인 함수 대신 직접 학습하는 방식입니다.
        self.PE = nn.Parameter(0.04*torch.rand(seq_patch_num, d_model)-0.02)
        # 0과 1 사이의 균등 분포(torch.rand)를 따르는 무작위 숫자를 요소로 [패치 개수, 임베딩 차원] 모양의 행렬을 생성합니다.
        
        # --- 디코더 ---
        self.decoder = Decoder(seq_patch_num, d_model, n_heads, pred_patch_num, d_ff=d_ff, attn_dropout=attn_dropout, dropout=dropout,
                               QAM_start=QAM_start, QAM_end=QAM_end, res_attention=res_attention, n_layers=n_layers, store_attn=store_attn)
        # 실제 어텐션 연산을 수행할 트랜스포머 디코더를 초기화합니다.
        
    # 순전파 로직을 정의합니다.
    def forward(self, x) -> Tensor:
        # 입력 x의 형태: [배치 크기, 변수 수, 시퀀스 패치 수, 패치 길이]
        # 디코더에 입력하기 위한 seq_patch와 pred_patch의 형태: [배치 크기*변수 수, 시퀀스 패치 수, d_model]
        # 디코더의 출력 형태: [배치 크기, 변수 수, 예측 패치 수, d_model]        
        bs = x.shape[0]
        # 입력 텐서에서 배치 크기를 가져옵니다.
        n_vars = x.shape[1]
        # 입력 텐서에서 변수의 수를 가져옵니다.

        # --- 1. 디코더에 입력할입력 시퀀스 준비. seq_patch: [배치 크기*변수 수, 시퀀스 패치 수, d_model] 구하기---
        x = self.W_P(x) + self.PE
        # 입력 패치에 Positional Encoding을 추가해주는 과정이다.
        # 입력 패치 `x`를 선형 레이어 `W_P(patch_len을 d_model로 변환)`로 임베딩하면 결과 x의 형태: [배치 크기, 변수 수, 시퀀스 패치 수, d_model] 
        # PE의 모양: [시퀀스 패치 수, d_model]
        # 수학식: Enc(x) = W_p(x) + PE. 브로드캐스팅을 통해 PE를 결과 x의 시퀀스 패치수, d_model 차원에 더합니다.
        # 결과 x의 형태: [배치 크기, 변수 수, 시퀀스 패치 수, d_model]

        x = torch.reshape(x, (bs*n_vars, x.shape[2], x.shape[3]))
        # 배치 차원과 변수 차원을 하나로 합쳐서 디코더가 처리하기 용이한 형태로 만듭니다.
        # 결과 x의 형태: [배치 크기*변수 수, 시퀀스 패치 수, d_model]
        
        seq_patch = self.dropout(x)
        # 인코딩된 입력 패치에 드롭아웃을 적용합니다.
         
        # --- 2. 디코더에 입력할 더미 입력 준비. pred_patch: [배치 크기*변수 수, 시퀀스 패치 수, d_model] 구하기---
        dummies = self.W_P(self.dummies)
        # 더미 입력 `dummies` 또한 동일한 선형 레이어 `W_P`로 임베딩합니다.
        # 결과 dummies의 형태: (독립적일 때) [예측 패치 수, d_model]. 왜냐하면 self.dummies = nn.Parameter(0.5*torch.randn(pred_patch_num, patch_len))
        # 결과 dummies의 형태: (종속적일 때) [변수 수, 예측 패치 수, d_model]. 왜냐하면 self.dummies = nn.Parameter(0.5*torch.randn(c_in, pred_patch_num, patch_len))
        
        if self.independence:
            # 변수들이 독립적일 경우,
            pred_patch = dummies.unsqueeze(0).repeat(bs*n_vars, 1, 1)           
            # 공유된 더미 입력을 배치 내 모든 샘플 및 변수에 대해 동일하게 복제합니다.
            # dummies: [예측 패치 수, d_model] 에 unsqueeze(0)을 해서 [1, 예측 패치 수, d_model] 로 만듦.
            # .repeat(bs*n_vars,1,1) 으로 첫 번째 차원(방금 추가한 차원)을 bs*n_vars번 복제합니다.
            # 최종 형태: [bs*n_vars, pred_patch_num, d_model]
        else:
            # 변수들이 종속적일 경우,
            pred_patch = dummies.unsqueeze(0).repeat(bs, 1, 1, 1)                
            # 각 변수별 더미 입력을 배치 내 모든 샘플에 대해 복제합니다.
            # dummies: [변수 수, 예측 패치 수, d_model] 애 unsqueeze(0)을 해서 [1, 변수 수, 예측 패치 수, d_model] 로 만듦.
            # .repeat(bs, 1, 1, 1) 으로 첫 번째 차원(방금 추가한 차원)을 bs번 복제합니다.
            # 그 결과: [bs, n_vars, pred_patch_num, d_model]
            
            pred_patch = torch.reshape(pred_patch, (bs*n_vars, pred_patch.shape[2], pred_patch.shape[3]))
            # 그 후, 배치 차원(bs), 변수 차원(n_vars)을 하나로 합칩니다.
            # 결과 pred_patch의 형태: [bs*n_vars, pred_patch_num, d_model]
        
        # --- 3. 디코더 순전파 ---
        z = self.decoder(seq_patch, pred_patch)
        # 준비된 입력 패치(`seq_patch`)와 더미 입력(`pred_patch`)을 디코더에 전달합니다.
        # `pred_patch`가 쿼리(Query) 역할을, `seq_patch`가 키(Key)와 값(Value) 역할을 수행합니다.
        # 입력되는 seq_patch, pred_patch의 형태: [배치 크기*변수 수, 시퀀스 패치 수, d_model]
        # 결과 z의 형태: [배치 크기 * 변수 수, 예측 패치 수, d_model]

        z = torch.reshape(z, (-1, n_vars, z.shape[-2], z.shape[-1]))
        # 디코더의 출력을 다시 [배치 크기, 변수 수, 예측 패치 수, d_model] 형태로 복원합니다.
        # 결과 z의 형태: [배치 크기, 변수 수, 예측 패치 수, d_model]        
        return z       
            
# 트랜스포머 백본의 출력을 최종 예측 시계열 형태로 변환하는 프로젝션 클래스입니다.
class Projection(nn.Module):
    # 프로젝션 클래스의 생성자입니다.
    def __init__(self, d_model, patch_len):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        self.linear = nn.Linear(d_model,patch_len)
        # 트랜스포머의 은닉 상태 차원, 즉 임베딩 차원(`d_model`)을 패치 길이(`patch_len`)로 변환하는 선형 레이어를 정의합니다.
        self.flatten = nn.Flatten(start_dim = -2)
        # 예측 패치들을 하나의 연속된 시계열로 펼치기 위한 Flatten 레이어를 정의합니다.
        # `start_dim=-2`는 마지막에서 두 번째 차원부터 평탄화를 시작하도록 지정합니다.
        # [배치 크기, 변수 수, pred_patch_num, patch_len] -> [배치 크기, 변수 수, pred_patch_num * patch_len]
            
    # 프로젝션의 순전파 로직을 정의합니다.
    def forward(self, x):
        x = self.linear(x)
        # 입력 `x`를 선형 레이어에 통과시켜 차원을 변환합니다.
        # 입력 x의 형태 (디코더의 최종 출력): [배치 크기, 변수 수, 예측 패치 수, d_model]  
        # linear 층 거친 후 결과: [배치 크기, 변수 수, 예측 패치 수, patch_len]
        
        x = self.flatten(x)
        # 변환된 텐서를 평탄화하여 예측 패치들을 하나의 시계열로 만듭니다.
        # 결과: [배치 크기, 변수 수, 예측 패치 수 * patch_len]
        return x                            
            
# 여러 개의 디코더 레이어로 구성된 트랜스포머 디코더 클래스입니다.
class Decoder(nn.Module):
    # 디코더의 생성자입니다.
    def __init__(self, seq_patch_num, d_model, n_heads, pred_patch_num, d_ff=None, attn_dropout=0., dropout=0., QAM_start = 0.1, QAM_end =0.5,
                        res_attention=False, n_layers=1, store_attn=False):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        self.layers = nn.ModuleList([DecoderLayer(seq_patch_num, d_model, pred_patch_num, n_heads=n_heads, d_ff=d_ff, QAM_start=QAM_start, QAM_end=QAM_end,
                                                      attn_dropout=attn_dropout, dropout=dropout, res_attention=res_attention,
                                                      store_attn=store_attn) for i in range(n_layers)])
        # `n_layers` 개수만큼의 `DecoderLayer`를 `nn.ModuleList`로 묶어 관리합니다.
        
        self.res_attention = res_attention
        # 잔차 어텐션(residual attention) 메커니즘 사용 여부를 저장합니다.
        # 잔차 어텐션은 한 디코더 레이어에서 계산된 어텐션 스코어 맵(Attention Score Map)을 다음 레이어로 넘겨주어, 다음 레이어의 어텐션 스코어 맵에 더하는 기법입니다.
    # 디코더의 순전파 로직을 정의합니다.
    def forward(self, seq:Tensor, pred:Tensor):
        scores = None
        # 잔차 어텐션에서 이전 레이어의 어텐션 스코어를 전달하기 위한 변수를 초기화합니다.
        
        if self.res_attention:
            # 잔차 어텐션을 사용하는 경우,
            for mod in self.layers: seq, pred, scores = mod(seq, pred, prev=scores)
            # 각 디코더 레이어를 순회하며, 이전 레이어의 어텐션 스코어(`scores`)를 다음 레이어로 전달합니다.
            # 이는 어텐션 스코어를 누적하여 사용하는 방식입니다. score_l층 = score_{l-1층} + attn_score_l층
            return pred
            # 모든 레이어를 통과한 최종 예측 결과 `pred`를 반환합니다.
        
        else:
            # 잔차 어텐션을 사용하지 않는 경우,
            for mod in self.layers: seq, pred = mod(seq, pred)
            # 각 디코-더 레이어를 독립적으로 통과시킵니다.
            return pred
            # 모든 레이어를 통과한 최종 예측 결과 `pred`를 반환합니다.

# 트랜스포머 디코더의 단일 레이어를 정의하는 클래스입니다.
# 크로스-어텐션(Cross-Attention)과 피드포워드 네트워크(Feed-Forward Network)로 구성됩니다.
class DecoderLayer(nn.Module):
    # 디코더 레이어의 생성자입니다.
    def __init__(self, seq_patch_num, d_model, pred_patch_num, n_heads, d_ff=256, store_attn=False, QAM_start = 0.1, QAM_end =0.5,
                 attn_dropout=0, dropout=0., bias=True, res_attention=False):
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        assert not d_model%n_heads, f"d_model ({d_model}) must be divisible by n_heads ({n_heads})"
        # `d_model`이 `n_heads`로 나누어 떨어져야 멀티헤드 어텐션이 가능하므로, 이를 확인합니다.
        
        # --- 크로스-어텐션 블록 ---
        self.res_attention = res_attention
        # 잔차 어텐션 사용 여부를 저장합니다.
        self.cross_attn = _MultiheadAttention(d_model, n_heads, attn_dropout=attn_dropout, proj_dropout=dropout, res_attention=res_attention)
        # 멀티헤드 크로스-어텐션 모듈을 초기화한다.
            # attn_dropout: 멀티헤드 어텐션 레이어에서 어텐션 가중치를 계산한 직후, 일부 가중치를 무작위로 0으로 만드는 드롭아웃 비율이다. 특정 키에만 과도하게 의존하는 것을 방지한다.
            # proj_dropout: 멀티헤드 어텐션 레이어는 각 헤드의 예측 결과를 이어붙이고 출력한다. 이 출력벡터의 일부를 무작위로 0으로 만드는 드롭아웃 비율.
        self.dropout_attn = QueryAdaptiveMasking(dim=1, start_prob=QAM_start, end_prob=QAM_end)
        # 어텐션 출력에 적용할 Query-Adaptive Masking 레이어를 정의합니다.
        self.norm_attn = nn.LayerNorm(d_model)
        # 어텐션 블록의 잔차 연결(add) 후 적용될 레이어 정규화(Layer Normalization)를 정의합니다.
            # 하나의 패치를 구성하는 d_model 개의 특징들을 정규화한다는 의미. 그냥 패치 내부 정규화.
            # nn.LayerNorm: 잔차 연결 결과 벡터 (pred = pred + pred2)를 평균0, 표준편차1 이 되도록 정규화한다.
            # (그리고 학습가능한 감마, 베타 (둘 다 d_model 차원 벡터) 를 사용하여 정규화가 필요한지를 학습한다. 최종 출력 = 감마 * (정규화된 출력) + 베타)

        # --- 피드포워드 네트워크 블록 ---
        # 위치별 피드포워드 네트워크(FFN)를 `nn.Sequential`로 정의합니다.
        self.ffn = nn.Sequential(nn.Linear(d_model, d_ff, bias=bias), # 1. d_model -> d_ff 확장
                                GEGLU(),                             # 2. GEGLU 활성화 함수 (이 활성함수를 거친 직후 차원이 절반이 됨))
                                nn.Dropout(dropout),                 # 3. 드롭아웃
                                nn.Linear(d_ff//2, d_model, bias=bias)) # 4. d_ff/2 -> d_model 축소 
        self.dropout_ffn = QueryAdaptiveMasking(dim=1, start_prob=QAM_start, end_prob=QAM_end)
        # FFN 출력에 적용할 Query-Adaptive Masking 레이어를 정의합니다.
        self.norm_ffn = nn.LayerNorm(d_model)
        # FFN 블록의 잔차 연결(add) 후 적용될 레이어 정규화를 정의합니다.
        
        self.store_attn = store_attn
        # 어텐션 가중치를 시각화 등의 목적으로 저장할지 여부를 결정합니다.

    # 디코더 레이어의 순전파 로직을 정의합니다.
    def forward(self, seq:Tensor, pred:Tensor, prev=None) -> Tensor:
        # `pred`는 쿼리(Q), `seq`는 키(K)와 값(V)으로 사용됩니다.
        
        # --- 멀티헤드 크로스-어텐션 ---
        if self.res_attention:
            # 잔차 어텐션을 사용하는 경우,
            pred2, attn, scores = self.cross_attn(pred, seq, seq, prev)
            # 어텐션 모듈은 (출력, 어텐션 가중치, 어텐션 스코어)를 반환합니다.
        else:
            # 잔차 어텐션을 사용하지 않는 경우,
            pred2, attn = self.cross_attn(pred, seq, seq)
            # 어텐션 모듈은 (출력, 어텐션 가중치)를 반환합니다.
        if self.store_attn:
            # 어텐션 가중치를 저장하도록 설정된 경우,
            self.attn = attn
            # 계산된 어텐션 가중치를 `self.attn`에 저장합니다.
        
        # --- 첫 번째 Add & Norm ---
        pred = pred + self.dropout_attn(pred2)
        # 1번째 Residual Connection    
        # 어텐션 출력(`pred2`)에 드롭아웃을 적용하고, 이를 입력(`pred`)에 더합니다 (잔차 연결). 수학식: pred = pred + dropout(pred2)
        pred = self.norm_attn(pred)
        # 레이어 정규화를 적용합니다. 수학식: pred = LayerNorm(pred)
        
        # --- 피드포워드 네트워크 ---
        pred2 = self.ffn(pred)
        # 정규화된 결과를 FFN에 통과시킵니다.

        # --- 두 번째 Add & Norm ---
        pred = pred + self.dropout_ffn(pred2)  
        # 2번째 Residual Connection   
        # FFN의 출력(`pred2`)에 드롭아웃을 적용하고, 이를 FFN의 입력(`pred`)에 더합니다 (잔차 연결). 수학식: pred = pred + dropout(pred2)
        pred = self.norm_ffn(pred)
        # 레이어 정규화를 적용합니다. 수학식: pred = LayerNorm(pred)
        
        if self.res_attention: return seq, pred, scores
        # 잔차 어텐션을 사용하는 경우, 다음 레이어로 전달할 값들을 포함하여 반환합니다.
        else: return seq, pred
        # 그렇지 않은 경우, 업데이트된 `seq`와 `pred`만 반환합니다.

# 멀티헤드 어텐션 메커니즘을 구현한 내부 클래스입니다.
class _MultiheadAttention(nn.Module):
    # 멀티헤드 어텐션의 생성자입니다.
    def __init__(self, d_model, n_heads, res_attention=False, attn_dropout=0., proj_dropout=0., qkv_bias=True):
        """
        멀티헤드 어텐션 레이어
        입력 형태:
            Q: [배치 크기, 미래 예측구간 패치 수, d_model]
            K, V: [배치 크기, 입력 시퀀스 패치 수, d_model]
        """
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        d_h = d_model // n_heads
        # 각 어텐션 헤드의 차원을 계산합니다.
        self.scale = d_h**-0.5
        # 어텐션 스코어를 스케일링하기 위한 팩터입니다. d_k의 제곱근의 역수(1/sqrt(d_k))를 사용합니다.
        # 이는 Q, K 내적 값의 분산이 d_k에 비례하여 커지는 것을 방지하여, softmax 함수의 기울기 소실(gradient vanishing) 문제를 완화합니다.
        self.n_heads, self.d_h = n_heads, d_h
        # 헤드의 수와 각 헤드의 차원을 저장합니다.

        # 입력받는 Q, K, V의 `d_model`차원을 `n_heads * d_h`으로 변환하는 선형 레이어들을 정의합니다.
        # 이때 d_model = n_heads * d_h 라서 왜 하는지 헷갈릴 수 있다.
        # 이 과정은 입력 벡터 Q, K, V를 멀티헤드 어텐션 레이어가 처리하기에 적합하도록 살짝 바꿔주는 정도로 이해하자.
        self.W_Q = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)
        self.W_K = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)
        self.W_V = nn.Linear(d_model, d_h * n_heads, bias=qkv_bias)

        self.res_attention = res_attention
        # 잔차 어텐션 사용 여부를 저장합니다.
        self.attn_dropout = nn.Dropout(attn_dropout)
        # 계산된 어텐션 가중치에 적용될 드롭아웃 레이어를 정의합니다.
        
        # n_heads * d_h 차원에서 같은 값인 d_model차원으로 변환시킨 후, 드롭아웃을 적용하는 출력 레이어를 정의합니다.
        self.to_out = nn.Sequential(nn.Linear(n_heads * d_h, d_model), nn.Dropout(proj_dropout))

    # 멀티헤드 어텐션의 순전파 로직을 정의합니다.
    def forward(self, Q:Tensor, K:Tensor, V:Tensor, prev=None):
        bs = Q.size(0)
        # 입력 쿼리(Q)에서 배치 크기를 가져옵니다.
        
        # --- Q, K, V 선형 변환 및 헤드 분할 ---
        # 
        # Q: [배치 크기, 미래 예측구간 패치 수, d_model]
        # K, V: [배치 크기, 입력 시퀀스 패치 수, d_model]
        
        # Q, K, V를 각각의 선형 레이어 W_Q, W_K, W_V에 통과시켜서 다음 값을 얻는다.
        # self.W_Q(Q): [배치 크기, 미래 예측구간 패치 수, n_heads * d_h]
        # self.W_k(k), self.W_v(v): [배치 크기, 입력 시퀀스 패치 수, n_heads * d_h]
        
        q_s = self.W_Q(Q).view(bs, -1, self.n_heads, self.d_h)     
        k_s = self.W_K(K).view(bs, -1, self.n_heads, self.d_h) 
        v_s = self.W_V(V).view(bs, -1, self.n_heads, self.d_h) 
        # 이후 'view`를 통해 멀티 헤드로 나눕니다.
        # .view(bs, -1, self.n_heads, self.d_h): bs유지, 패치수 알아서, 마지막 차원이었던 n_heads * d_h 를 n_heads, d_h으로 쪼갬.
        # 즉 3차원에서 4차원으로 변형한 것이다.
        # 결과: [배치 크기, 패치 수, 멀티 헤드 개수, 헤드 차원]

        # --- 어텐션 스코어 계산 ---
        attn_scores = torch.einsum('bphd, bshd -> bphs', q_s, k_s) * self.scale
        # `torch.einsum` (아인슈타인 표기법)을 사용하여 Q와 K의 내적을 효율적으로 계산합니다.
        # b: batch, p: pred_patch(Q의 시퀀스 차원), s: sequence_patch(K의 시퀀스 차원), h: head, d: dimension
        # 첫 번째 입력 (bphd): q_s 텐서의 차원을 나타냅니다.
        # 두 번째 입력 (bshd): k_s 텐서의 차원을 나타냅니다.
        # 출력 (-> bphs): 연산 결과로 나올 텐서의 차원을 나타냅니다.
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
        # 'bphs, bshd -> bphd'는 (배치, 쿼리패치, 헤드, 키패치)와 (배치, 키패치, 헤드, 차원)을 곱해 (배치, 쿼리패치, 헤드, 차원)을 만듭니다.
        # 이는 Attention Weight * V 와 동일한 연산입니다.
        output = output.contiguous().view(bs, -1, self.n_heads*self.d_h)
        # 나뉘었던 헤드들을 `contiguous()`와 `view`를 통해 다시 하나의 텐서로 합칩니다.
        # 이전: [bs, p, n_heads, d_h]
        # 결과: [bs, p, n_heads*d_h]
        
        output = self.to_out(output)
        # 최종 출력 레이어를 통과시킵니다.
        # n_heads * d_h 차원에서 같은 값인 d_model차원으로 변환시킨 후, 드롭아웃을 적용하는 출력 레이어를 정의합니다.

        if self.res_attention: return output, attn_weights, attn_scores
        # 잔차 어텐션을 사용하는 경우, (출력, 어텐션 가중치, 어텐션 스코어)를 모두 반환합니다.
        else: return output, attn_weights
        # 그렇지 않은 경우, (출력, 어텐션 가중치)만 반환합니다.

# 전체 모델을 구성하고 순전파를 정의하는 메인 클래스입니다.
# 하이퍼파라미터를 인자로 받아 `Model_backbone`을 초기화하고 데이터의 입출력 형식을 관리합니다.
class Model(nn.Module):
    # 전체 모델의 생성자입니다.
    def __init__(self, args, **kwargs):
        
        super().__init__()
        # `nn.Module`의 생성자를 호출합니다.
        
        # --- 하이퍼파라미터 로드 ---
        # `args` 객체로부터 모델 구성에 필요한 모든 하이퍼파라미터를 가져옵니다.
        c_in = args.dec_in               # 입력 변수(채널)의 수
        seq_len = args.seq_len           # 입력 시퀀스의 길이
        self.pred_len = args.pred_len    # 예측할 시퀀스의 길이
        n_layers = args.d_layers         # 트랜스포머 디코더의 레이어 수
        n_heads = args.n_heads           # 멀티헤드 어텐션의 헤드 수
        d_model = args.d_model           # 모델의 은닉 상태 차원
        d_ff = args.d_ff                 # 피드포워드 네트워크의 내부 차원
        dropout = args.dropout           # 드롭아웃 비율
        independence = args.query_independence # 변수 독립성 여부
        patch_len = args.patch_len       # 각 패치의 길이
        stride = args.stride             # 패치 추출 시의 보폭
        padding_patch = args.padding_patch # 패딩 전략
        store_attn = args.store_attn     # 어텐션 가중치 저장 여부
        QAM_start = args.QAM_start       # QAM 시작 확률
        QAM_end = args.QAM_end           # QAM 끝 확률

        # 로드한 하이퍼파라미터들을 사용하여 `Model_backbone`을 초기화합니다.
        self.model = Model_backbone(c_in=c_in, seq_len = seq_len, pred_len=self.pred_len, patch_len=patch_len, stride=stride, n_layers=n_layers, 
                                    d_model=d_model, n_heads=n_heads, d_ff=d_ff, dropout=dropout,independence=independence, 
                                    store_attn=store_attn, padding_patch = padding_patch, QAM_start=QAM_start, QAM_end=QAM_end, **kwargs)

    
    # 전체 모델의 순전파 로직을 정의합니다.
    def forward(self, x): # 입력 x의 일반적인 형태: [배치 크기, 입력 길이, 채널 수]
        x = x.permute(0,2,1)
        # `permute`를 사용하여 입력 텐서의 차원 순서를 CATS 모델의 입력 형식에 맞게끔 [배치, 채널, 길이]로 변경합니다.
        # 이는 모델 내부에서 채널별로 연산을 처리하기 용이하게 하기 위함입니다.
        x = self.model(x)
        # 백본 모델에 입력을 통과시켜 예측을 수행합니다.
        x = x.permute(0,2,1)
        # 모델의 출력 또한 [배치, 채널, 길이] 형태이므로, 다시 [배치, 길이, 채널] 형태로 되돌립니다.
        return x[:,:self.pred_len,:]
        # 최종적으로, 예측된 시퀀스에서 필요한 `pred_len` 만큼의 길이만 잘라서 반환합니다.