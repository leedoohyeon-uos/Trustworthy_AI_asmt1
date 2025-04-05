# test.py
# FGSM 및 PGD 적대적 공격 구현 및 테스트
# 작성자: 이두현
# 작성일: 2025-04-04

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
import os
import time
import argparse

# GPU 메모리 할당 설정
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

# 랜덤 시드 고정 (재현 가능성)
np.random.seed(42)
tf.random.set_seed(42)

# CIFAR-10 클래스 이름
cifar10_classes = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]

def load_cifar10_data():
    """CIFAR-10 데이터셋을 로드하고 전처리하는 함수
    
    Returns:
        x_train, y_train, x_test, y_test: 전처리된 학습 및 테스트 데이터
    """
    # CIFAR-10 데이터셋 로드
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()
    
    # 데이터 형태 출력
    print('학습 데이터 형태:', x_train.shape)
    print('학습 라벨 형태:', y_train.shape)
    print('테스트 데이터 형태:', x_test.shape)
    print('테스트 라벨 형태:', y_test.shape)
    
    # 0~1 범위로 정규화
    x_train = x_train.astype('float32') / 255.0
    x_test = x_test.astype('float32') / 255.0
    
    # 라벨 원-핫 인코딩
    y_train_categorical = to_categorical(y_train, 10)
    y_test_categorical = to_categorical(y_test, 10)
    
    # 라벨 형태 변경 (2D -> 1D)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)
    
    return x_train, y_train_categorical, y_train, x_test, y_test_categorical, y_test

def fgsm_targeted(model, x, target, eps=0.03):
    """Targeted FGSM (Fast Gradient Sign Method) 공격 구현
    
    타겟 클래스를 향해 예측을 변경하도록 입력 이미지를 수정하는 공격 기법
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        target: 타겟 클래스 (정수)
        eps: 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 타겟 원-핫 인코딩 - 배치 차원 추가
    target_one_hot = tf.one_hot([target], 10)  # [1, 10] 형태로 변환
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)  # 이미지에 대한 그래디언트를 계산하기 위해 감시
        prediction = model(x_adv)
        # 타겟 클래스의 확률을 최대화하는 방향으로 손실함수 정의 (음수화)
        loss = -tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
    
    # 입력에 대한 손실의 그레디언트 계산
    gradient = tape.gradient(loss, x_adv)
    
    # FGSM 방식으로 적대적 샘플 생성
    signed_grad = tf.sign(gradient)  # 그래디언트의 부호만 사용
    x_adv = x_adv - eps * signed_grad  # 타겟 방향으로 이동하기 위해 빼기 사용
    
    # [0, 1] 범위로 클리핑
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv.numpy()

def fgsm_untargeted(model, x, label, eps=0.03):
    """Untargeted FGSM (Fast Gradient Sign Method) 공격 구현
    
    현재 클래스에서 멀어지도록 입력 이미지를 수정하는 공격 기법
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        label: 원본 이미지의 실제 라벨 (정수)
        eps: 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 라벨 원-핫 인코딩 - 배치 차원 추가
    label_one_hot = tf.one_hot([label], 10)  # [1, 10] 형태로 변환
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)  # 이미지에 대한 그래디언트를 계산하기 위해 감시
        prediction = model(x_adv)
        # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
        loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
    
    # 입력에 대한 손실의 그레디언트 계산
    gradient = tape.gradient(loss, x_adv)
    
    # FGSM 방식으로 적대적 샘플 생성
    signed_grad = tf.sign(gradient)  # 그래디언트의 부호만 사용
    x_adv = x_adv + eps * signed_grad  # 원래 클래스에서 멀어지기 위해 더하기 사용
    
    # [0, 1] 범위로 클리핑
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv.numpy()

def pgd_targeted(model, x, target, k=10, eps=0.03, eps_step=0.01):
    """Targeted PGD (Projected Gradient Descent) 공격 구현
    
    여러 단계에 걸쳐 타겟 클래스를 향해 예측을 변경하는 반복적 공격 기법
    
    Args:
        model: 타겟 모델
        x: 원본 이미지 (단일 이미지 또는 배치)
        target: 타겟 클래스 (정수)
        k: 반복 횟수
        eps: 최대 공격 강도
        eps_step: 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = x.copy()
    
    # 원본 이미지 범위 설정 (eps 반경 내)
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 타겟 원-핫 인코딩 - 배치 차원 추가
    target_one_hot = tf.one_hot([target], 10)  # [1, 10] 형태로 변환
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 타겟 클래스의 확률을 최대화하는 방향으로 손실함수 정의
            loss = -tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # 그레디언트 방향으로 한 스텝 이동
        signed_grad = tf.sign(gradient)
        x_adv = x_adv - eps_step * signed_grad.numpy()  # 타겟 방향으로 이동하기 위해 빼기 사용
        
        # 원본 이미지의 eps 범위 내로 제한 (투영)
        x_adv = np.clip(x_adv, x_min, x_max)
        
        # [0, 1] 범위로 추가 클리핑
        x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv

def pgd_untargeted(model, x, label, k=10, eps=0.03, eps_step=0.01):
    """Untargeted PGD (Projected Gradient Descent) 공격 구현
    
    여러 단계에 걸쳐 현재 클래스에서 멀어지도록 예측을 변경하는 반복적 공격 기법
    
    Args:
        model: 타겟 모델
        x: 원본 이미지 (단일 이미지 또는 배치)
        label: 원본 이미지의 실제 라벨 (정수)
        k: 반복 횟수
        eps: 최대 공격 강도
        eps_step: 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = x.copy()
    
    # 원본 이미지 범위 설정 (eps 반경 내)
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 라벨 원-핫 인코딩 - 배치 차원 추가
    label_one_hot = tf.one_hot([label], 10)  # [1, 10] 형태로 변환
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
            loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # 그레디언트 방향으로 한 스텝 이동
        signed_grad = tf.sign(gradient)
        x_adv = x_adv + eps_step * signed_grad.numpy()  # 원래 클래스에서 멀어지기 위해 더하기 사용
        
        # 원본 이미지의 eps 범위 내로 제한 (투영)
        x_adv = np.clip(x_adv, x_min, x_max)
        
        # [0, 1] 범위로 추가 클리핑
        x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv

def pgd_untargeted(model, x, label, k=10, eps=0.03, eps_step=0.01):
    """Untargeted PGD (Projected Gradient Descent) 공격 구현
    
    여러 단계에 걸쳐 현재 클래스에서 멀어지도록 예측을 변경하는 반복적 공격 기법
    
    Args:
        model: 타겟 모델
        x: 원본 이미지 (단일 이미지 또는 배치)
        label: 원본 이미지의 실제 라벨 (정수)
        k: 반복 횟수
        eps: 최대 공격 강도
        eps_step: 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = x.copy()
    
    # 원본 이미지 범위 설정 (eps 반경 내)
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 라벨 원-핫 인코딩
    label_one_hot = tf.one_hot(label, 10)
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
            loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # 그레디언트 방향으로 한 스텝 이동
        signed_grad = tf.sign(gradient)
        x_adv = x_adv + eps_step * signed_grad.numpy()  # 원래 클래스에서 멀어지기 위해 더하기 사용
        
        # 원본 이미지의 eps 범위 내로 제한 (투영)
        x_adv = np.clip(x_adv, x_min, x_max)
        
        # [0, 1] 범위로 추가 클리핑
        x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv

def visualize_attack_results(model, x_orig, x_adv, true_label, attack_name, target=None):
    """적대적 공격 결과를 시각화하는 함수
    
    Args:
        model: 타겟 모델
        x_orig: 원본 이미지
        x_adv: 적대적 이미지
        true_label: 실제 라벨
        attack_name: 공격 이름
        target: 타겟 클래스 (targeted 공격에만 사용)
    """
    # 예측
    orig_pred = model.predict(np.expand_dims(x_orig, axis=0))[0]
    adv_pred = model.predict(np.expand_dims(x_adv, axis=0))[0]
    
    orig_class = np.argmax(orig_pred)
    adv_class = np.argmax(adv_pred)
    
    # 이미지 시각화
    plt.figure(figsize=(12, 4))
    
    # 원본 이미지
    plt.subplot(1, 3, 1)
    plt.imshow(x_orig)
    plt.title(f"Original: {cifar10_classes[orig_class]}\n(True: {cifar10_classes[true_label]})")
    plt.axis('off')
    
    # 적대적 이미지
    plt.subplot(1, 3, 2)
    plt.imshow(x_adv)
    if target is not None:
        plt.title(f"Adversarial: {cifar10_classes[adv_class]}\n(Target: {cifar10_classes[target]})")
    else:
        plt.title(f"Adversarial: {cifar10_classes[adv_class]}")
    plt.axis('off')
    
    # 차이 이미지
    plt.subplot(1, 3, 3)
    diff = np.abs(x_adv - x_orig)
    # 차이를 강조하기 위해 스케일 조정
    plt.imshow(diff * 5)  # 차이를 5배 강조
    plt.title("Perturbation (x5)")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.axis('off')
    
    plt.suptitle(f"{attack_name} Attack", fontsize=16)
    plt.tight_layout()
    
    # 결과 저장 디렉토리 생성
    os.makedirs("attack_results", exist_ok=True)
    plt.savefig(f"attack_results/{attack_name}_example.png")
    
    # 예측 결과 출력
    print(f"\n{attack_name} Attack 결과")
    print(f"원본 이미지 예측: {cifar10_classes[orig_class]} (신뢰도: {orig_pred[orig_class]:.4f})")
    if target is not None:
        print(f"타겟 클래스: {cifar10_classes[target]}")
    print(f"적대적 이미지 예측: {cifar10_classes[adv_class]} (신뢰도: {adv_pred[adv_class]:.4f})")
    print(f"원본 이미지 라벨: {cifar10_classes[true_label]}")
    
    # 공격 성공 여부
    if target is not None:
        success = (adv_class == target)
        print(f"공격 성공 여부 (타겟={cifar10_classes[target]}): {'성공' if success else '실패'}")
    else:
        success = (adv_class != true_label)
        print(f"공격 성공 여부 (원본 클래스에서 이탈): {'성공' if success else '실패'}")

    # L2 거리 계산
    l2_distance = np.sqrt(np.sum((x_orig - x_adv) ** 2))
    print(f"L2 거리: {l2_distance:.6f}")
    
    # L∞ 거리 계산
    linf_distance = np.max(np.abs(x_orig - x_adv))
    print(f"L∞ 거리: {linf_distance:.6f}")

def run_attack(attack_type, model, x, true_label, target=None, eps=0.03, k=10, eps_step=0.01):
    """지정된 공격 유형에 따라 적대적 공격을 실행하는 함수
    
    Args:
        attack_type: 공격 유형 (targeted_fgsm, untargeted_fgsm, targeted_pgd, untargeted_pgd)
        model: 타겟 모델
        x: 원본 이미지
        true_label: 실제 라벨
        target: 타겟 클래스 (targeted 공격에만 사용)
        eps: 공격 강도
        k: PGD 반복 횟수
        eps_step: PGD 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    start_time = time.time()
    
    if attack_type == "targeted_fgsm":
        if target is None:
            raise ValueError("Targeted 공격에는 타겟 클래스가 필요합니다.")
        x_adv = fgsm_targeted(model, np.expand_dims(x, axis=0), target, eps=eps)[0]
        attack_name = "Targeted_FGSM"
    
    elif attack_type == "untargeted_fgsm":
        x_adv = fgsm_untargeted(model, np.expand_dims(x, axis=0), true_label, eps=eps)[0]
        attack_name = "Untargeted_FGSM"
    
    elif attack_type == "targeted_pgd":
        if target is None:
            raise ValueError("Targeted 공격에는 타겟 클래스가 필요합니다.")
        x_adv = pgd_targeted(model, np.expand_dims(x, axis=0), target, k=k, eps=eps, eps_step=eps_step)[0]
        attack_name = "Targeted_PGD"
    
    elif attack_type == "untargeted_pgd":
        x_adv = pgd_untargeted(model, np.expand_dims(x, axis=0), true_label, k=k, eps=eps, eps_step=eps_step)[0]
        attack_name = "Untargeted_PGD"
    
    else:
        raise ValueError(f"지원되지 않는 공격 유형: {attack_type}")
    
    print(f"공격 소요 시간: {time.time() - start_time:.2f}초")
    
    # 결과 시각화
    visualize_attack_results(model, x, x_adv, true_label, attack_name, target)
    
    return x_adv

def evaluate_model_robustness(model, x_test, y_test, num_samples=100, eps=0.03, eps_step=0.01, k=10):
    """모델의 적대적 공격에 대한 강건성을 평가하는 함수
    
    Args:
        model: 평가할 모델
        x_test: 테스트 데이터
        y_test: 테스트 라벨
        num_samples: 평가에 사용할 샘플 수
        eps: 공격 강도
        eps_step: PGD 각 단계의 공격 강도
        k: PGD 반복 횟수
    """
    if num_samples > len(x_test):
        num_samples = len(x_test)
    
    # 랜덤 샘플 선택
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    x_subset = x_test[indices]
    y_subset = y_test[indices]
    
    # 원본 정확도 계산
    predictions = np.argmax(model.predict(x_subset), axis=1)
    original_accuracy = np.mean(predictions == y_subset)
    
    # 각 공격 방법에 대한 정확도 저장
    attack_accuracies = {}
    
    # 1. Untargeted FGSM
    adv_samples = []
    for i in range(num_samples):
        adv_sample = fgsm_untargeted(model, np.expand_dims(x_subset[i], axis=0), y_subset[i], eps=eps)
        adv_samples.append(adv_sample[0])
    
    adv_samples = np.array(adv_samples)
    predictions = np.argmax(model.predict(adv_samples), axis=1)
    attack_accuracies["Untargeted FGSM"] = np.mean(predictions == y_subset)
    
    # 2. Untargeted PGD
    adv_samples = []
    for i in range(num_samples):
        adv_sample = pgd_untargeted(model, np.expand_dims(x_subset[i], axis=0), y_subset[i], k=k, eps=eps, eps_step=eps_step)
        adv_samples.append(adv_sample[0])
    
    adv_samples = np.array(adv_samples)
    predictions = np.argmax(model.predict(adv_samples), axis=1)
    attack_accuracies["Untargeted PGD"] = np.mean(predictions == y_subset)
    
    # 결과 출력
    print("\n============= 모델 강건성 평가 =============")
    print(f"샘플 수: {num_samples}")
    print(f"원본 정확도: {original_accuracy:.4f}")
    for attack_name, accuracy in attack_accuracies.items():
        print(f"{attack_name} 공격 후 정확도: {accuracy:.4f}")
        print(f"{attack_name} 공격 성공률: {1 - accuracy:.4f}")
    
    # 결과 시각화
    plt.figure(figsize=(10, 6))
    accuracies = [original_accuracy] + list(attack_accuracies.values())
    attack_names = ["원본"] + list(attack_accuracies.keys())
    
    plt.bar(attack_names, accuracies)
    plt.ylabel("정확도")
    plt.title("적대적 공격에 대한 모델 강건성")
    plt.ylim([0, 1])
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # 값 표시
    for i, v in enumerate(accuracies):
        plt.text(i, v + 0.02, f"{v:.3f}", ha='center')
    
    plt.tight_layout()
    os.makedirs("attack_results", exist_ok=True)
    plt.savefig("attack_results/model_robustness.png")

def main():
    """메인 함수"""
    # 명령행 인자 파싱
    parser = argparse.ArgumentParser(description="적대적 공격 테스트")
    parser.add_argument("--model", type=str, default="cifar10_resnet50_model.h5", help="모델 파일 경로")
    parser.add_argument("--attack", type=str, default="all", help="공격 유형 (targeted_fgsm, untargeted_fgsm, targeted_pgd, untargeted_pgd, all)")
    parser.add_argument("--eps", type=float, default=0.03, help="공격 강도 (엡실론)")
    parser.add_argument("--steps", type=int, default=10, help="PGD 반복 횟수")
    parser.add_argument("--step_size", type=float, default=0.005, help="PGD 단계 크기")
    parser.add_argument("--evaluate", action="store_true", help="모델 강건성 평가 실행")
    parser.add_argument("--samples", type=int, default=100, help="강건성 평가에 사용할 샘플 수")
    
    args = parser.parse_args()
    
    # 데이터 로드
    print("CIFAR-10 데이터 로드 중...")
    _, _, _, x_test, _, y_test = load_cifar10_data()
    
    # 모델 로드
    print(f"모델 로드 중: {args.model}")
    try:
        model = load_model(args.model)
    except:
        print(f"오류: 모델 파일을 찾을 수 없습니다: {args.model}")
        print("main_CIFAR_10.py를 먼저 실행하여 모델을 학습하세요.")
        return
    
    # 모델 평가
    loss, accuracy = model.evaluate(x_test, to_categorical(y_test, 10), verbose=1)
    print(f"테스트 정확도: {accuracy:.4f}")
    
    # 모델 강건성 평가
    if args.evaluate:
        evaluate_model_robustness(model, x_test, y_test, num_samples=args.samples, 
                                 eps=args.eps, eps_step=args.step_size, k=args.steps)
        return
    
    # 공격 대상 이미지 선택
    idx = np.random.randint(0, len(x_test))
    x = x_test[idx]
    true_label = y_test[idx]
    
    # 타겟 클래스 선택 (원래 클래스와 다르게)
    available_targets = [i for i in range(10) if i != true_label]
    target = np.random.choice(available_targets)
    
    print(f"\n선택된 이미지 인덱스: {idx}")
    print(f"실제 라벨: {true_label} ({cifar10_classes[true_label]})")
    print(f"타겟 라벨: {target} ({cifar10_classes[target]})")
    
    # 원본 이미지 예측
    orig_pred = model.predict(np.expand_dims(x, axis=0))[0]
    orig_class = np.argmax(orig_pred)
    print(f"원본 이미지 예측: {orig_class} ({cifar10_classes[orig_class]}, 신뢰도: {orig_pred[orig_class]:.4f})")
    
    # 지정된 공격 또는 모든 공격 실행
    if args.attack == "all" or args.attack == "targeted_fgsm":
        print("\nTargeted FGSM 공격 수행 중...")
        run_attack("targeted_fgsm", model, x, true_label, target=target, eps=args.eps)
    
    if args.attack == "all" or args.attack == "untargeted_fgsm":
        print("\nUntargeted FGSM 공격 수행 중...")
        run_attack("untargeted_fgsm", model, x, true_label, eps=args.eps)
    
    if args.attack == "all" or args.attack == "targeted_pgd":
        print("\nTargeted PGD 공격 수행 중...")
        run_attack("targeted_pgd", model, x, true_label, target=target, 
                  eps=args.eps, k=args.steps, eps_step=args.step_size)
    
    if args.attack == "all" or args.attack == "untargeted_pgd":
        print("\nUntargeted PGD 공격 수행 중...")
        run_attack("untargeted_pgd", model, x, true_label, 
                  eps=args.eps, k=args.steps, eps_step=args.step_size)
    
    print("\n적대적 공격 테스트 완료!")

if __name__ == "__main__":
    main()