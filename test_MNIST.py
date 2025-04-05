# test_MNSIT.py
# MNIST 모델 및 적대적 공격 테스트 코드
# 작성일: 2025-04-04

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import os
from tensorflow.keras.models import load_model
from main import (load_mnist_data, train_model, fgsm_targeted, fgsm_untargeted, 
                 pgd_targeted, pgd_untargeted, visualize_attack_results)

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

def test_model_accuracy():
    """모델의 정확도를 테스트하는 함수"""
    print("="*50)
    print("모델 정확도 테스트")
    print("="*50)
    
    # MNIST 데이터 로드
    _, _, x_test, t_test = load_mnist_data()
    
    # 모델 로드 또는 학습
    model_path = "mnist_cnn_model.h5"
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        print("저장된 모델이 없습니다. 새로 학습합니다.")
        x_train, t_train, _, _ = load_mnist_data()
        model = train_model(x_train, t_train, x_test, t_test, model_path)
    
    # 테스트 데이터에 대한 정확도 평가
    loss, accuracy = model.evaluate(x_test, t_test)
    print(f"테스트 데이터 정확도: {accuracy:.4f}")
    
    return model, x_test, t_test

def test_fgsm_attacks(model, x_test, t_test, num_samples=5):
    """FGSM 공격을 테스트하는 함수"""
    print("\n" + "="*50)
    print("FGSM 공격 테스트")
    print("="*50)
    
    # 결과 저장 디렉토리 생성
    os.makedirs("attack_results", exist_ok=True)
    
    # 테스트할 샘플 인덱스
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    success_targeted = 0
    success_untargeted = 0
    
    for i, idx in enumerate(indices):
        print(f"\n샘플 {i+1}/{num_samples} 테스트 중...")
        
        # 원본 이미지와 라벨
        x = x_test[idx]
        true_label = t_test[idx]
        
        # 타겟 클래스 선택 (원래 클래스와 다르게)
        available_targets = [i for i in range(10) if i != true_label]
        target = np.random.choice(available_targets)
        
        print(f"실제 라벨: {true_label}, 타겟 라벨: {target}")
        
        # 원본 이미지 예측
        orig_pred = model.predict(np.expand_dims(x, axis=0))[0]
        orig_class = np.argmax(orig_pred)
        
        # 1. Targeted FGSM 공격
        print("Targeted FGSM 공격 수행 중...")
        x_adv_fgsm_t = fgsm_targeted(model, np.expand_dims(x, axis=0), target, eps=0.3)
        
        # 공격 결과 확인
        adv_pred = model.predict(x_adv_fgsm_t)[0]
        adv_class = np.argmax(adv_pred)
        
        if adv_class == target:
            success_targeted += 1
            print(f"Targeted FGSM 공격 성공: {orig_class} -> {adv_class} (타겟: {target})")
        else:
            print(f"Targeted FGSM 공격 실패: {orig_class} -> {adv_class} (타겟: {target})")
        
        # 결과 시각화
        visualize_attack_results(model, x, x_adv_fgsm_t[0], true_label, f"Targeted_FGSM_Sample{i+1}", target)
        
        # 2. Untargeted FGSM 공격
        print("Untargeted FGSM 공격 수행 중...")
        x_adv_fgsm_u = fgsm_untargeted(model, np.expand_dims(x, axis=0), true_label, eps=0.3)
        
        # 공격 결과 확인
        adv_pred = model.predict(x_adv_fgsm_u)[0]
        adv_class = np.argmax(adv_pred)
        
        if adv_class != true_label:
            success_untargeted += 1
            print(f"Untargeted FGSM 공격 성공: {orig_class} -> {adv_class} (원래: {true_label})")
        else:
            print(f"Untargeted FGSM 공격 실패: {orig_class} -> {adv_class} (원래: {true_label})")
        
        # 결과 시각화
        visualize_attack_results(model, x, x_adv_fgsm_u[0], true_label, f"Untargeted_FGSM_Sample{i+1}")
    
    print("\nFGSM 공격 성공률:")
    print(f"Targeted FGSM: {success_targeted/num_samples:.2%}")
    print(f"Untargeted FGSM: {success_untargeted/num_samples:.2%}")

def test_pgd_attacks(model, x_test, t_test, num_samples=5):
    """PGD 공격을 테스트하는 함수"""
    print("\n" + "="*50)
    print("PGD 공격 테스트")
    print("="*50)
    
    # 결과 저장 디렉토리 생성
    os.makedirs("attack_results", exist_ok=True)
    
    # 테스트할 샘플 인덱스
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    success_targeted = 0
    success_untargeted = 0
    
    for i, idx in enumerate(indices):
        print(f"\n샘플 {i+1}/{num_samples} 테스트 중...")
        
        # 원본 이미지와 라벨
        x = x_test[idx]
        true_label = t_test[idx]
        
        # 타겟 클래스 선택 (원래 클래스와 다르게)
        available_targets = [i for i in range(10) if i != true_label]
        target = np.random.choice(available_targets)
        
        print(f"실제 라벨: {true_label}, 타겟 라벨: {target}")
        
        # 원본 이미지 예측
        orig_pred = model.predict(np.expand_dims(x, axis=0))[0]
        orig_class = np.argmax(orig_pred)
        
        # 3. Targeted PGD 공격
        print("Targeted PGD 공격 수행 중...")
        x_adv_pgd_t = pgd_targeted(model, np.expand_dims(x, axis=0), target, k=10, eps=0.3, eps_step=0.03)
        
        # 공격 결과 확인
        adv_pred = model.predict(x_adv_pgd_t)[0]
        adv_class = np.argmax(adv_pred)
        
        if adv_class == target:
            success_targeted += 1
            print(f"Targeted PGD 공격 성공: {orig_class} -> {adv_class} (타겟: {target})")
        else:
            print(f"Targeted PGD 공격 실패: {orig_class} -> {adv_class} (타겟: {target})")
        
        # 결과 시각화
        visualize_attack_results(model, x, x_adv_pgd_t[0], true_label, f"Targeted_PGD_Sample{i+1}", target)
        
        # 4. Untargeted PGD 공격
        print("Untargeted PGD 공격 수행 중...")
        x_adv_pgd_u = pgd_untargeted(model, np.expand_dims(x, axis=0), true_label, k=10, eps=0.3, eps_step=0.03)
        
        # 공격 결과 확인
        adv_pred = model.predict(x_adv_pgd_u)[0]
        adv_class = np.argmax(adv_pred)
        
        if adv_class != true_label:
            success_untargeted += 1
            print(f"Untargeted PGD 공격 성공: {orig_class} -> {adv_class} (원래: {true_label})")
        else:
            print(f"Untargeted PGD 공격 실패: {orig_class} -> {adv_class} (원래: {true_label})")
        
        # 결과 시각화
        visualize_attack_results(model, x, x_adv_pgd_u[0], true_label, f"Untargeted_PGD_Sample{i+1}")
    
    print("\nPGD 공격 성공률:")
    print(f"Targeted PGD: {success_targeted/num_samples:.2%}")
    print(f"Untargeted PGD: {success_untargeted/num_samples:.2%}")

def test_different_epsilons(model, x_test, t_test, num_samples=3):
    """다양한 epsilon 값에 따른 공격 효과를 테스트하는 함수"""
    print("\n" + "="*50)
    print("다양한 epsilon 값 테스트")
    print("="*50)
    
    # 테스트할 샘플 인덱스
    indices = np.random.choice(len(x_test), num_samples, replace=False)
    
    # 테스트할 epsilon 값
    epsilons = [0.05, 0.1, 0.2, 0.3, 0.4]
    
    # 결과 저장을 위한 딕셔너리
    results = {
        'targeted_fgsm': [0] * len(epsilons),
        'untargeted_fgsm': [0] * len(epsilons),
        'targeted_pgd': [0] * len(epsilons),
        'untargeted_pgd': [0] * len(epsilons)
    }
    
    for i, idx in enumerate(indices):
        print(f"\n샘플 {i+1}/{num_samples} 테스트 중...")
        
        # 원본 이미지와 라벨
        x = x_test[idx]
        true_label = t_test[idx]
        
        # 타겟 클래스 선택 (원래 클래스와 다르게)
        available_targets = [i for i in range(10) if i != true_label]
        target = np.random.choice(available_targets)
        
        print(f"실제 라벨: {true_label}, 타겟 라벨: {target}")
        
        for j, eps in enumerate(epsilons):
            print(f"\nEpsilon = {eps} 테스트 중...")
            
            # 1. Targeted FGSM
            x_adv = fgsm_targeted(model, np.expand_dims(x, axis=0), target, eps=eps)
            pred = np.argmax(model.predict(x_adv)[0])
            if pred == target:
                results['targeted_fgsm'][j] += 1
            
            # 2. Untargeted FGSM
            x_adv = fgsm_untargeted(model, np.expand_dims(x, axis=0), true_label, eps=eps)
            pred = np.argmax(model.predict(x_adv)[0])
            if pred != true_label:
                results['untargeted_fgsm'][j] += 1
            
            # 3. Targeted PGD
            x_adv = pgd_targeted(model, np.expand_dims(x, axis=0), target, k=10, eps=eps, eps_step=eps/10)
            pred = np.argmax(model.predict(x_adv)[0])
            if pred == target:
                results['targeted_pgd'][j] += 1
            
            # 4. Untargeted PGD
            x_adv = pgd_untargeted(model, np.expand_dims(x, axis=0), true_label, k=10, eps=eps, eps_step=eps/10)
            pred = np.argmax(model.predict(x_adv)[0])
            if pred != true_label:
                results['untargeted_pgd'][j] += 1
    
    # 결과 시각화
    plt.figure(figsize=(12, 8))
    
    for i, attack_type in enumerate(results.keys()):
        success_rates = [count / num_samples for count in results[attack_type]]
        plt.subplot(2, 2, i+1)
        plt.plot(epsilons, success_rates, marker='o')
        plt.title(attack_type.replace('_', ' ').title())
        plt.xlabel('Epsilon')
        plt.ylabel('Success Rate')
        plt.grid(True)
        plt.ylim(0, 1.05)
    
    plt.tight_layout()
    plt.savefig("attack_results/epsilon_comparison.png")
    plt.show()
    
    # 결과 출력
    print("\n다양한 epsilon 값에 따른 공격 성공률:")
    for j, eps in enumerate(epsilons):
        print(f"Epsilon = {eps}:")
        for attack_type in results.keys():
            print(f"  {attack_type.replace('_', ' ').title()}: {results[attack_type][j]/num_samples:.2%}")

def main():
    """테스트 메인 함수"""
    print("MNIST 적대적 공격 테스트 시작")
    
    # 1. 모델 정확도 테스트
    model, x_test, t_test = test_model_accuracy()
    
    # 2. FGSM 공격 테스트
    test_fgsm_attacks(model, x_test, t_test, num_samples=3)
    
    # 3. PGD 공격 테스트
    test_pgd_attacks(model, x_test, t_test, num_samples=3)
    
    # 4. 다양한 epsilon 값에 따른 효과 테스트
    test_different_epsilons(model, x_test, t_test, num_samples=2)
    
    print("\n모든 테스트가 완료되었습니다!")

if __name__ == "__main__":
    main()