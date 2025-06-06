import numpy as np
import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map

# 1. Параметрүүд
MAP_SIZE = 7
HOLE_PROB = 0.8
SEED = 42  # Тогтвортой үр дүн гаргахын тулд seed тохируулав

# Q-Learning-ийн параметрүүд
NUM_EPISODES = 50000  # Q-Learning-д илүү олон эпизод хэрэгтэй байж болно
LEARNING_RATE = 0.9  # (Alpha) Q-утгыг хэр хурдан шинэчлэх вэ
DISCOUNT_FACTOR = 0.9  # (Gamma) Ирээдүйн шагналын ач холбогдол
EPSILON_START = 1.0  # Эхлэх үеийн илрүүлэх магадлал
EPSILON_DECAY_RATE = 0.00005  # Эпизод бүрт epsilon-ийг хэр багасгах вэ
MIN_EPSILON = 0.01  # Epsilon-ийн хамгийн бага утга

MAX_STEPS_PER_EPISODE = 200  # Нэг эпизодын хамгийн их алхам


def train_q_learning(env):
    """Q-Learning алгоритмаар бодлого сургах функц."""
    np.random.seed(SEED)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    # Q-хүснэгтийг 0-ээр эхлүүлэх (state x action хэмжээтэй)
    q_table = np.zeros((n_states, n_actions))

    epsilon = EPSILON_START  # Эхлэх epsilon-ийг тохируулах

    print(f"Q-Learning сургалт эхэллээ. Эпизодын тоо: {NUM_EPISODES}")
    print(f"Сургах map: {env.unwrapped.desc}")  # Train хийсэн map-ыг харуулна

    for episode in range(NUM_EPISODES):
        state, _ = (
            env.reset()
        )  # Эпизод бүрт seed-г өөрчилж санамсаргүй байдлыг нэмэгдүүлнэ
        done = False

        for step in range(MAX_STEPS_PER_EPISODE):
            # Epsilon-greedy үйлдлийн сонголт
            if np.random.uniform(0, 1) < epsilon:
                action = (
                    env.action_space.sample()
                )  # Илрүүлэх (Explore): Санамсаргүй үйлдлийг сонгох
            else:
                action = np.argmax(
                    q_table[state, :]
                )  # Ашиглах (Exploit): Одоогийн хамгийн сайн үйлдлийг сонгох

            # Үйлдлийг гүйцэтгэж, орчноос үр дүнг авах
            new_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Q-Learning-ийн шинэчлэх дүрэм
            # Q(s,a) = Q(s,a) + alpha * [r + gamma * max(Q(s',a')) - Q(s,a)]
            q_table[state, action] = q_table[state, action] + LEARNING_RATE * (
                reward
                + DISCOUNT_FACTOR * np.max(q_table[new_state, :])
                - q_table[state, action]
            )
            state = new_state  # Төлөвийг шинэчлэх
            if done:
                break  # Эпизод дууссан бол зогсоох

        # Epsilon-ийг багасгах (аажимдаа илрүүлэх байдлыг бууруулах)
        epsilon = max(MIN_EPSILON, epsilon - EPSILON_DECAY_RATE)

        # Прогрессийг хэвлэх (заавал биш)
        if (episode + 1) % (NUM_EPISODES // 10) == 0:  # 10%-ийн давтамжтай хэвлэх
            print(f"  Эпизод: {episode + 1}/{NUM_EPISODES}, Epsilon: {epsilon:.4f}")

    print("Q-Learning сургалт дууслаа.")
    return q_table


def q_table_to_policy(q_table):
    """Q-хүснэгтээс хамгийн сайн бодлогыг гаргах (Greedy Policy)."""
    # Төлөв бүрт хамгийн их Q-утгатай үйлдлийг сонгоно
    policy = np.argmax(q_table, axis=1)
    return policy


if __name__ == "__main__":
    # Санамсаргүй FrozenLake map үүсгэх (run_frozenlake.py-тэй ижил map ашиглах ёстой)
    random_map = generate_random_map(size=MAP_SIZE, p=HOLE_PROB, seed=SEED)

    # Орчин үүсгэх (сургалтын үед render_mode-гүйгээр хурдан ажиллана)
    env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=False)

    # Q-Learning-ээр сургах
    q_table_trained = train_q_learning(env)

    # Q-хүснэгтээс хамгийн сайн бодлогыг гаргах
    best_policy = q_table_to_policy(q_table_trained)

    print(f"\nСуралцсан хамгийн сайн бодлого (Q-Learning-аас): {best_policy}")

    # Хадгалах
    np.save("best_policy.npy", best_policy)
    print("→ best_policy.npy файлд хадгаллаа.")

    env.close()  # Орчинг хаах
