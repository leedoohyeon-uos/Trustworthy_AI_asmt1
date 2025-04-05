# main_MNSIT.py
# MNIST 데이터셋을 사용한 CNN 모델 학습 및 적대적 공격 구현
# 작성자: 이두현
# 날짜: 2025-04-04

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import Dense, Input, Activation, Flatten, Conv2D, MaxPooling2D, BatchNormalization, Dropout
import os
import time

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

# 유틸리티 함수: 이미지와 예측 결과 시각화
def plot_image(i, predictions_array, true_label, img, class_names):
    """이미지와 예측 결과를 시각화하는 함수
    
    Args:
        i: 인덱스
        predictions_array: 예측 확률 배열
        true_label: 실제 라벨
        img: 이미지 데이터
        class_names: 클래스 이름 리스트
    """
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    img = np.squeeze(img)
    plt.imshow(img, cmap=plt.cm.binary)
    
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'
    
    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                100*np.max(predictions_array),
                                class_names[true_label]),
                                color=color)

def plot_value_array(i, predictions_array, true_label):
    """예측 확률 분포를 시각화하는 함수
    
    Args:
        i: 인덱스
        predictions_array: 예측 확률 배열
        true_label: 실제 라벨
    """
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')

def load_mnist_data():
    """MNIST 데이터셋을 로드하고 전처리하는 함수
    
    Returns:
        x_train, t_train, x_test, t_test: 전처리된 학습 및 테스트 데이터
    """
    # MNIST 데이터셋 로드
    (x_train, t_train), (x_test, t_test) = mnist.load_data()
    
    # 0~1 범위로 정규화
    x_train = (x_train/255.).astype('float32')
    x_test = (x_test/255.).astype('float32')
    
    # CNN 입력용 형태로 변환 (채널 차원 추가)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    print('학습 데이터 형태:', x_train.shape)
    print('학습 라벨 형태:', t_train.shape)
    print('테스트 데이터 형태:', x_test.shape)
    print('테스트 라벨 형태:', t_test.shape)
    
    return x_train, t_train, x_test, t_test

def conv_op(_xin, _iter=3, nband=[256, 64, 16], filt_size=[(3, 3), (3, 3), (3, 3)],
           rate=[(1, 1), (1, 1), (1, 1)], k_init='he_normal',
           pad=['same', 'same', 'same'], f_act=['relu', 'relu', 'relu'], drop_rate=0.3):
    """CNN 모델의 컨볼루션 블록을 생성하는 함수
    
    Args:
        _xin: 입력 텐서
        _iter: 반복 횟수
        nband: 각 층의 필터 수
        filt_size: 각 층의 필터 크기
        rate: 각 층의 dilation rate
        k_init: 커널 초기화 방법
        pad: 각 층의 패딩 방법
        f_act: 각 층의 활성화 함수
        drop_rate: 드롭아웃 비율
        
    Returns:
        conv: 컨볼루션 블록의 출력 텐서
    """
    conv = _xin

    for i in range(_iter):
        conv = Conv2D(nband[i], filt_size[i], kernel_initializer=k_init, 
                      dilation_rate=rate[i], padding=pad[i])(conv)
        conv = BatchNormalization()(conv)
        conv = Activation(f_act[i])(conv)
        if i != _iter-1:
            if drop_rate != 0:
                conv = Dropout(drop_rate)(conv)

    return conv

def create_cnn_model(shape=(28, 28, 1), nclass=10):
    """CNN 모델을 생성하는 함수
    
    Args:
        shape: 입력 이미지 형태
        nclass: 출력 클래스 수
        
    Returns:
        model: 컴파일된 CNN 모델
    """
    _in = Input(shape=shape)

    # 인코딩 부분
    x = conv_op(_in)
    x = MaxPooling2D((2, 2))(x)  # 28 -> 14
    x = conv_op(x)
    x = MaxPooling2D((2, 2))(x)  # 14 -> 7
    x = conv_op(x)

    x = Flatten()(x)
    x = Dense(1024)(x)
    x = Activation('relu')(x)
    x = Dense(128)(x)
    x = Activation('relu')(x)
    x = Dense(nclass)(x)
    _out = Activation('softmax')(x)

    model = Model(inputs=[_in], outputs=[_out])
    
    return model

def train_model(x_train, t_train, x_test, t_test, model_path="mnist_cnn_model.h5"):
    """CNN 모델을 학습하는 함수
    
    Args:
        x_train: 학습 데이터
        t_train: 학습 라벨
        x_test: 테스트 데이터
        t_test: 테스트 라벨
        model_path: 모델 저장 경로
        
    Returns:
        model: 학습된 모델
    """
    # 모델 파라미터 설정
    lr = 0.001
    batch_size = 128
    epochs = 15  # 학습 에폭 수 설정
    
    # 모델 생성 및 컴파일
    model = create_cnn_model()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=['accuracy']
    )
    
    # 모델 구조 출력
    model.summary()
    
    # 콜백 설정
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=5, verbose=1),
        keras.callbacks.ModelCheckpoint(model_path, save_best_only=True, monitor='val_accuracy', verbose=1),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=1)
    ]
    
    # 모델 학습
    print("모델 학습을 시작합니다...")
    history = model.fit(
        x_train, t_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, t_test),
        callbacks=callbacks
    )
    
    # 학습 과정 시각화
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Training and Validation Accuracy')
    
    plt.tight_layout()
    plt.savefig("training_history.png")
    plt.show()
    
    # 테스트 데이터에 대한 평가
    test_loss, test_acc = model.evaluate(x_test, t_test, verbose=2)
    print(f"\n테스트 정확도: {test_acc:.4f}")
    
    # 모델 저장
    model.save(model_path)
    print(f"모델이 {model_path}에 저장되었습니다.")
    
    return model

def fgsm_targeted(model, x, target, eps=0.3):
    """Targeted FGSM (Fast Gradient Sign Method) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        target: 타겟 클래스 (정수)
        eps: 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 타겟 원-핫 인코딩 - 차원 추가 (배치 차원)
    target_one_hot = tf.one_hot(target, 10)
    target_one_hot = tf.expand_dims(target_one_hot, 0)  # 차원 추가 (1, 10)
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        prediction = model(x_adv)
        # 타겟 클래스의 확률을 최대화하는 방향으로 손실함수 정의
        loss = -tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
    
    # 입력에 대한 손실의 그레디언트 계산
    gradient = tape.gradient(loss, x_adv)
    
    # FGSM 방식으로 적대적 샘플 생성
    signed_grad = tf.sign(gradient)
    x_adv = x_adv - eps * signed_grad  # 타겟 방향으로 이동하기 위해 빼기 사용
    
    # [0, 1] 범위로 클리핑
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv.numpy()

def fgsm_untargeted(model, x, label, eps=0.3):
    """Untargeted FGSM (Fast Gradient Sign Method) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        label: 원본 이미지의 실제 라벨 (정수)
        eps: 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = tf.convert_to_tensor(x, dtype=tf.float32)
    
    # 라벨 원-핫 인코딩 - 차원 추가 (배치 차원)
    label_one_hot = tf.one_hot(label, 10)
    label_one_hot = tf.expand_dims(label_one_hot, 0)  # 차원 추가 (1, 10)
    
    with tf.GradientTape() as tape:
        tape.watch(x_adv)
        prediction = model(x_adv)
        # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
        loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
    
    # 입력에 대한 손실의 그레디언트 계산
    gradient = tape.gradient(loss, x_adv)
    
    # FGSM 방식으로 적대적 샘플 생성
    signed_grad = tf.sign(gradient)
    x_adv = x_adv + eps * signed_grad  # 원래 클래스에서 멀어지기 위해 더하기 사용
    
    # [0, 1] 범위로 클리핑
    x_adv = tf.clip_by_value(x_adv, 0, 1)
    
    return x_adv.numpy()

def pgd_targeted(model, x, target, k=10, eps=0.3, eps_step=0.01):
    """Targeted PGD (Projected Gradient Descent) 공격 구현
    
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
    
    # 원본 이미지 범위 설정
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 타겟 원-핫 인코딩 - 차원 추가 (배치 차원)
    target_one_hot = tf.one_hot(target, 10)
    target_one_hot = tf.expand_dims(target_one_hot, 0)  # 차원 추가 (1, 10)
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 타겟 클래스의 확률을 최대화하는 방향으로 손실함수 정의
            loss = -tf.keras.losses.categorical_crossentropy(target_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # FGSM 방식으로 적대적 샘플 업데이트
        signed_grad = tf.sign(gradient)
        x_adv = x_adv - eps_step * signed_grad.numpy()  # 타겟 방향으로 이동하기 위해 빼기 사용
        
        # 원본 이미지의 eps 범위 내로 제한
        x_adv = np.clip(x_adv, x_min, x_max)
        
        # [0, 1] 범위로 추가 클리핑
        x_adv = np.clip(x_adv, 0, 1)
    
    return x_adv

def pgd_untargeted(model, x, label, k=10, eps=0.3, eps_step=0.01):
    """Untargeted PGD (Projected Gradient Descent) 공격 구현
    
    Args:
        model: 타겟 모델
        x: 원본 이미지
        label: 원본 이미지의 실제 라벨 (정수)
        k: 반복 횟수
        eps: 최대 공격 강도
        eps_step: 각 단계의 공격 강도
        
    Returns:
        x_adv: 적대적 이미지
    """
    x_adv = x.copy()
    
    # 원본 이미지 범위 설정
    x_min = np.clip(x - eps, 0, 1)
    x_max = np.clip(x + eps, 0, 1)
    
    # 라벨 원-핫 인코딩 - 차원 추가 (배치 차원)
    label_one_hot = tf.one_hot(label, 10)
    label_one_hot = tf.expand_dims(label_one_hot, 0)  # 차원 추가 (1, 10)
    
    for i in range(k):
        x_adv_tensor = tf.convert_to_tensor(x_adv, dtype=tf.float32)
        
        with tf.GradientTape() as tape:
            tape.watch(x_adv_tensor)
            prediction = model(x_adv_tensor)
            # 실제 클래스의 확률을 최소화하는 방향으로 손실함수 정의
            loss = tf.keras.losses.categorical_crossentropy(label_one_hot, prediction)
        
        # 입력에 대한 손실의 그레디언트 계산
        gradient = tape.gradient(loss, x_adv_tensor)
        
        # FGSM 방식으로 적대적 샘플 업데이트
        signed_grad = tf.sign(gradient)
        x_adv = x_adv + eps_step * signed_grad.numpy()  # 원래 클래스에서 멀어지기 위해 더하기 사용
        
        # 원본 이미지의 eps 범위 내로 제한
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
    orig_pred = model.predict(np.expand_dims(x_orig, axis=0))[0]
    adv_pred = model.predict(np.expand_dims(x_adv, axis=0))[0]
    
    orig_class = np.argmax(orig_pred)
    adv_class = np.argmax(adv_pred)
    
    plt.figure(figsize=(12, 5))
    
    # 원본 이미지 표시
    plt.subplot(1, 3, 1)
    plt.imshow(np.squeeze(x_orig), cmap='gray')
    plt.title(f"Original: {orig_class} (True: {true_label})")
    plt.axis('off')
    
    # 적대적 이미지 표시
    plt.subplot(1, 3, 2)
    plt.imshow(np.squeeze(x_adv), cmap='gray')
    if target is not None:
        plt.title(f"Adversarial: {adv_class} (Target: {target})")
    else:
        plt.title(f"Adversarial: {adv_class}")
    plt.axis('off')
    
    # 차이 이미지 표시
    plt.subplot(1, 3, 3)
    diff = np.abs(x_adv - x_orig)
    plt.imshow(np.squeeze(diff), cmap='hot')
    plt.title("Perturbation")
    plt.colorbar()
    plt.axis('off')
    
    plt.suptitle(f"{attack_name} Attack", fontsize=16)
    plt.tight_layout()
    
    # 결과 저장
    os.makedirs("attack_results", exist_ok=True)
    plt.savefig(f"attack_results/{attack_name}_example.png")
    plt.show()
    
    print(f"\n{attack_name} Attack 결과")
    print(f"원본 이미지 예측: {orig_class} (신뢰도: {orig_pred[orig_class]:.4f})")
    if target is not None:
        print(f"타겟 클래스: {target}")
    print(f"적대적 이미지 예측: {adv_class} (신뢰도: {adv_pred[adv_class]:.4f})")
    print(f"원본 이미지 라벨: {true_label}")
    
    # 원본 및 적대적 이미지의 예측 확률 분포 비교
    plt.figure(figsize=(10, 5))
    
    plt.subplot(1, 2, 1)
    plt.bar(range(10), orig_pred)
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.title("Original Prediction")
    
    plt.subplot(1, 2, 2)
    plt.bar(range(10), adv_pred)
    plt.xticks(range(10))
    plt.ylim([0, 1])
    plt.title("Adversarial Prediction")
    
    plt.tight_layout()
    plt.savefig(f"attack_results/{attack_name}_predictions.png")
    plt.show()

def main():
    """메인 함수"""
    print("MNIST 데이터 로드 중...")
    x_train, t_train, x_test, t_test = load_mnist_data()
    
    print("\nCNN 모델 학습 중...")
    model = train_model(x_train, t_train, x_test, t_test)
    
    print("\n학습된 모델로 테스트 데이터 평가 중...")
    loss, accuracy = model.evaluate(x_test, t_test, verbose=0)
    print(f"Test accuracy: {accuracy:.4f}")
    
    print("\n적대적 공격 예시 생성 중...")
    
    # 테스트 이미지 선택
    idx = np.random.randint(0, len(x_test))
    x = x_test[idx]
    true_label = t_test[idx]
    
    # 타겟 클래스 선택 (원래 클래스와 다르게)
    available_targets = [i for i in range(10) if i != true_label]
    target = np.random.choice(available_targets)
    
    print(f"\n선택된 이미지 인덱스: {idx}")
    print(f"실제 라벨: {true_label}")
    print(f"타겟 라벨: {target}")
    
    # 원본 이미지 예측
    orig_pred = model.predict(np.expand_dims(x, axis=0))[0]
    orig_class = np.argmax(orig_pred)
    print(f"원본 이미지 예측: {orig_class} (신뢰도: {orig_pred[orig_class]:.4f})")
    
    # 1. Targeted FGSM 공격
    print("\nTargeted FGSM 공격 수행 중...")
    x_adv_fgsm_t = fgsm_targeted(model, np.expand_dims(x, axis=0), target, eps=0.2)
    visualize_attack_results(model, x, x_adv_fgsm_t[0], true_label, "Targeted FGSM", target)
    
    # 2. Untargeted FGSM 공격
    print("\nUntargeted FGSM 공격 수행 중...")
    x_adv_fgsm_u = fgsm_untargeted(model, np.expand_dims(x, axis=0), true_label, eps=0.2)
    visualize_attack_results(model, x, x_adv_fgsm_u[0], true_label, "Untargeted FGSM")
    
    # 3. Targeted PGD 공격
    print("\nTargeted PGD 공격 수행 중...")
    x_adv_pgd_t = pgd_targeted(model, np.expand_dims(x, axis=0), target, k=10, eps=0.2, eps_step=0.025)
    visualize_attack_results(model, x, x_adv_pgd_t[0], true_label, "Targeted PGD", target)
    
    # 4. Untargeted PGD 공격
    print("\nUntargeted PGD 공격 수행 중...")
    x_adv_pgd_u = pgd_untargeted(model, np.expand_dims(x, axis=0), true_label, k=10, eps=0.2, eps_step=0.025)
    visualize_attack_results(model, x, x_adv_pgd_u[0], true_label, "Untargeted PGD")
    
    print("\n적대적 공격 시연 완료!")

if __name__ == "__main__":
    main()