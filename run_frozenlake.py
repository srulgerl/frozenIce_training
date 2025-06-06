import numpy as np
import gymnasium as gym
import time
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# 1. Параметрүүд (train-тэй ижил байх ёстой, учир нь ижил map-ыг ашиглана)
MAP_SIZE = 7
HOLE_PROB = 0.8
SEED = 42  # Train хийсэн map-тай ижил map үүсгэхийн тулд ижил seed ашиглана


def play_with_policy(env, policy, delay=0.5):
    """Хамгийн сайн бодлогоор тоглох функц."""
    state, _ = env.reset()
    env.render()

    print("\nТоглоом эхэллээ. Enter дарж эхлүүлнэ үү...")

    for step in range(100):
        action = policy[state]
        print(f"Алхам {step+1}: Төлөв = {state}, Үйлдэл = {action}")
        state, reward, terminated, truncated, _ = env.step(action)
        time.sleep(delay)
        env.render()

        if terminated or truncated:
            if reward == 1:
                print("🎉 Амжилттай зорилгод хүрлээ!")
            else:
                print("💥 Унасан...")
            break
    else:  # 100 алхамд зорилгод хүрээгүй бол
        print("💡 100 алхамд зорилгод хүрч чадсангүй.")


if __name__ == "__main__":
    # Санамсаргүй FrozenLake map үүсгэх (train_frozenlake_qlearning.py-тэй ижил параметрээр)
    # ЗӨВХӨН ЭНЭ MAP-ЫГ TRAIN ХИЙСЭН БОДЛОГОТОЙ АЖИЛЛАНА!
    random_map = generate_random_map(size=MAP_SIZE, p=HOLE_PROB, seed=SEED)

    print(f"Тоглох map: {random_map}")  # Тоглох map-ыг харуулна

    # 2. Policy-г файлнаас унших
    try:
        best_policy = np.load("best_policy.npy")
        print("best_policy.npy файлыг амжилттай уншлаа.")
    except FileNotFoundError:
        print("Алдаа: best_policy.npy файл олдсонгүй.")
        print(
            "Эхлээд 'train_frozenlake_qlearning.py' скриптийг ажиллуулж бодлого сургана уу."
        )
        exit()  # Хэрэв файл байхгүй бол програмыг зогсооно.

    # 3. Орчин (render_mode="human" нь тоглоомыг харуулах горим)
    env = gym.make(
        "FrozenLake-v1", desc=random_map, is_slippery=False, render_mode="human"
    )

    play_with_policy(env, best_policy)

    env.close()  # Орчинг хаах
