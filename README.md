# FrozenLake Q-Learning Project

Энэ төсөл нь OpenAI Gymnasium-ийн FrozenLake орчин дээр Q-Learning алгоритм ашиглан бодлого сургаж, хадгалсан бодлогоор тоглох жишээ юм.

## Файлын бүтэц

- `train_frozenlake_qlearning.py`  
  Q-Learning алгоритмаар бодлого сургаж, хамгийн сайн бодлогыг `best_policy.npy` файлд хадгална.

- `run_frozenlake.py`  
  Сурсан бодлогыг (`best_policy.npy`) ачаалж, FrozenLake орчинд тоглож үзнэ.

- `best_policy.npy`  
  Сурсан хамгийн сайн бодлогын numpy файл.

- `.gitignore`  
  Git-д оруулах шаардлагагүй файлуудыг заана.

## Ашиглах заавар

1. **Хамааралтай сангуудыг суулгах:**
   ```sh
   pip install gymnasium numpy
   ```

2. **Q-Learning сургалт хийх:**
   ```sh
   python train_frozenlake_qlearning.py
   ```
   Энэ нь `best_policy.npy` файлыг үүсгэнэ.

3. **Сурсан бодлогоор тоглох:**
   ```sh
   python run_frozenlake.py
   ```

## Тайлбар

- `MAP_SIZE`, `HOLE_PROB`, `SEED` зэрэг параметрүүдийг хоёр скриптэд ижилхэн тохируулсан байх ёстой.
- `train_frozenlake_qlearning.py` скриптээр сурсан бодлого л `run_frozenlake.py`-д ажиллана.
- Орчин нь `is_slippery=False` тохиргоотой, санамсаргүй map-тай.
