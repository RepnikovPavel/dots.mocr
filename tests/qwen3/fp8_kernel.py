import sys
import importlib

def test_fp8_kernel():
    print("=" * 60)
    print("ШАГ 1: Проверка наличия пакета 'kernels'")
    print("=" * 60)
    try:
        import kernels
        print(f"[OK] Пакет 'kernels' установлен. Версия: {getattr(kernels, '__version__', 'неизвестна')}")
        print(f"      Расположение: {kernels.__file__}")
    except ImportError:
        print("[FAIL] Пакет 'kernels' НЕ установлен!")
        print("       Именно это вызывает возврат None из lazy_load_kernel.")
        print("       РЕШЕНИЕ: pip install kernels")
        return

    print("\n" + "=" * 60)
    print("ШАГ 2: Проверка функции lazy_load_kernel")
    print("=" * 60)
    try:
        from transformers.integrations.hub_kernels import lazy_load_kernel
        print("[OK] Функция lazy_load_kernel импортирована")
    except ImportError as e:
        print(f"[FAIL] Не удалось импортировать lazy_load_kernel: {e}")
        return

    print("\n" + "=" * 60)
    print("ШАГ 3: Вызов lazy_load_kernel('finegrained-fp8')")
    print("=" * 60)
    
    # Включаем детализацию ошибок, если пакет kernels поддерживает логирование
    import logging
    logging.basicConfig(level=logging.DEBUG)
    
    try:
        kernel = lazy_load_kernel("finegrained-fp8")
        
        if kernel is None:
            print("[FAIL] lazy_load_kernel вернула None!")
            print("       Это означает, что ядро не найдено на Hub или")
            print("       произошла ошибка JIT-компиляции Triton (Triton 3.6.0 несовместим).")
            print("\n       РЕШЕНИЕ:")
            print("       1) pip install --upgrade kernels")
            print("       2) Если не помогает: pip install triton==3.1.0")
        else:
            print(f"[OK] Ядро загружено! Тип: {type(kernel)}")
            
            print("\n" + "=" * 60)
            print("ШАГ 4: Проверка наличия нужных функций в ядре")
            print("=" * 60)
            required_funcs = [
                "w8a8_fp8_matmul",
                "fp8_act_quant", 
                "w8a8_fp8_matmul_batched",
                "w8a8_fp8_matmul_grouped"
            ]
            
            all_ok = True
            for func_name in required_funcs:
                has_it = hasattr(kernel, func_name)
                status = "[OK]" if has_it else "[FAIL]"
                print(f"  {status} {func_name}")
                if not has_it:
                    all_ok = False
            
            if all_ok:
                print("\n✅ ВСЕ ПРОВЕРКИ ПРОЙДЕНЫ. FP8 ядро полностью валидно.")
            else:
                print("\n❌ Ядро загружено, но часть функций отсутствует.")
                print("   Решение: pip install --upgrade kernels")
                
    except Exception as e:
        print(f"[FAIL] Исключение при загрузке ядра: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        print("\nЭто реальная ошибка компиляции. Решение:")
        print("pip install triton==3.1.0")


if __name__ == "__main__":
    test_fp8_kernel()