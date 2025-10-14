import matplotlib.pyplot as plt
import re
import os

def parse_accuracy_from_log(log_file_path):
    """
    로그 파일에서 'Test Accuracy' 값을 파싱하여 리스트로 반환합니다.
    """
    accuracies = []
    # 정규 표현식: "Test Accuracy: " 뒤에 오는 소수점 숫자를 찾습니다.
    # 예: "Test Accuracy: 85.34%" -> "85.34"
    accuracy_pattern = re.compile(r"Test Accuracy: (\d+\.\d+)\%")
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            for line in f:
                match = accuracy_pattern.search(line)
                if match:
                    # 찾은 문자열을 float 형태로 변환하여 리스트에 추가합니다.
                    accuracies.append(float(match.group(1)))
    except FileNotFoundError:
        print(f"오류: 로그 파일 '{log_file_path}'를 찾을 수 없습니다. 경로를 확인해주세요.")
        return None
    except Exception as e:
        print(f"파일 처리 중 오류 발생: {e}")
        return None
        
    return accuracies

def plot_graphs(log_files, labels):
    """
    여러 로그 파일의 정확도 데이터를 하나의 그래프에 시각화합니다.
    """
    plt.style.use('seaborn-v0_8-whitegrid') # 그래프 스타일 설정
    fig, ax = plt.subplots(figsize=(12, 8)) # 그래프 크기 설정

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'] # 그래프 색상
    markers = ['o', 's', 'D', '^'] # 데이터 포인트 마커

    for i, (log_file, label) in enumerate(zip(log_files, labels)):
        print(f"'{log_file}' 파일 처리 중...")
        accuracies = parse_accuracy_from_log(log_file)
        
        if accuracies:
            epochs = range(1, len(accuracies) + 1)
            ax.plot(epochs, accuracies, 
                    marker=markers[i % len(markers)], 
                    linestyle='-', 
                    color=colors[i % len(colors)], 
                    label=label)
            print(f" -> {len(accuracies)}개의 데이터 포인트를 찾았습니다.")
        else:
            print(f" -> '{log_file}'에서 정확도 데이터를 찾지 못했습니다.")

    # 그래프 제목 및 축 레이블 설정
    ax.set_title('Test Accuracy Comparison per Epoch', fontsize=16)
    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Test Accuracy (%)', fontsize=12)
    
    # 범례 표시
    if ax.has_data():
        ax.legend(fontsize=10)
    
    # 그래프 표시
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    # 분석할 로그 파일 경로 리스트
    # ★★★ 사용자의 환경에 맞게 경로를 수정해주세요 ★★★
    log_file_paths = [
        '/home/hannibal/cau_cats/log/Refined_mix inch3 ep100 max0.5/log_20251014_030136.log',
        '/home/hannibal/cau_cats/log/tap_new inch3 ep100 max0.5/log_20251014_070324.log'
    ]
    
    # 각 그래프에 표시될 이름 (라벨)
    graph_labels = [
        'Refined_mix',
        'tap_new'
    ]
    
    plot_graphs(log_file_paths, graph_labels)

