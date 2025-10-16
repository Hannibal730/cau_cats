--opts 뒤에 덮어쓰고 싶은 변수들을 .으로 경로를 지정하고 =으로 값을 할당하면 됩니다. 여러 변수를 변경하고 싶을 경우, 공백으로 구분하여 나열하면 됩니다.

예시 1: 에포크(epoch)와 학습률(learning rate) 변경

bash
python run.py --opts training.epochs=50 training.lr=0.0005