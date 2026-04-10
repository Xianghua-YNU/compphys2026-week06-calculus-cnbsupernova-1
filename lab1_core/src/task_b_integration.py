import math
from typing import Callable


def debye_integrand(x: float) -> float:
    """
    Debye 积分核函数: x^4 * e^x / (e^x - 1)^2
    
    当 x 接近 0 时，函数值为 0（通过极限分析可得）。
    """
    if abs(x) < 1e-12:
        return 0.0
    ex = math.exp(x)
    return (x**4) * ex / ((ex - 1.0) ** 2)


def trapezoid_composite(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    复合梯形积分 (Composite Trapezoidal Rule)
    
    将区间 [a, b] 分成 n 等份，步长 h = (b-a)/n，
    积分近似为: h/2 * [f(a) + 2*Σf(x_i) + f(b)]
    
    Parameters:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        n: 分段数（正整数）
    
    Returns:
        积分近似值
    """
    if n <= 0:
        raise ValueError("n must be positive")
    
    h = (b - a) / n
    result = 0.5 * (f(a) + f(b))
    
    for i in range(1, n):
        x = a + i * h
        result += f(x)
    
    result *= h
    return result


def simpson_composite(f: Callable[[float], float], a: float, b: float, n: int) -> float:
    """
    复合 Simpson 积分 (Composite Simpson's Rule)
    
    要求 n 为偶数，将区间 [a, b] 分成 n 等份，步长 h = (b-a)/n，
    每两个小区间使用 Simpson 公式，系数模式为 1, 4, 2, 4, 2, ..., 4, 1
    
    Parameters:
        f: 被积函数
        a: 积分下限
        b: 积分上限
        n: 分段数（偶数）
    
    Returns:
        积分近似值
    """
    if n <= 0:
        raise ValueError("n must be positive")
    if n % 2 != 0:
        raise ValueError("n must be even for Simpson's rule")
    
    h = (b - a) / n
    result = f(a) + f(b)
    
    # 奇数索引点 (i=1,3,5...) 系数为 4
    for i in range(1, n, 2):
        x = a + i * h
        result += 4 * f(x)
    
    # 偶数索引点 (i=2,4,6...) 系数为 2
    for i in range(2, n, 2):
        x = a + i * h
        result += 2 * f(x)
    
    result *= h / 3
    return result


def debye_integral(T: float, theta_d: float = 428.0, method: str = "simpson", n: int = 200) -> float:
    """
    计算 Debye 积分 I(y) = ∫[0,y] (x^4 e^x)/(e^x-1)^2 dx
    其中 y = theta_d / T
    
    这是 Debye 模型中计算热容的关键积分，描述晶格振动对热容的贡献。
    
    Parameters:
        T: 温度 (K)，必须为正数
        theta_d: Debye 温度 (K)，默认 428 K（铜的典型值）
        method: 积分方法，"trapezoid" 或 "simpson"
        n: 分段数（Simpson 方法要求偶数）
    
    Returns:
        积分值 I(y)
    """
    if T <= 0:
        raise ValueError("Temperature T must be positive")
    
    y = theta_d / T
    
    if method == "trapezoid":
        return trapezoid_composite(debye_integrand, 0.0, y, n)
    elif method == "simpson":
        return simpson_composite(debye_integrand, 0.0, y, n)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'trapezoid' or 'simpson'.")


# 比较两种方法在相同 n 下的误差差异
if __name__ == "__main__":
    print("Debye 热容积分 - 数值方法比较")
    print("=" * 70)
    
    theta_d = 428.0  # Debye 温度 (K)
    temperatures = [50, 100, 200, 300, 428, 1000]  # 不同温度点
    n = 100  # 分段数
    
    print(f"\n1. 不同温度下的积分值比较 (n={n}):")
    print("-" * 70)
    print(f"{'T (K)':<10} {'y=θD/T':<10} {'Trapezoid':<18} {'Simpson':<18} {'|Diff|':<15}")
    print("-" * 70)
    
    for T in temperatures:
        y = theta_d / T
        trap_val = debye_integral(T, theta_d, method="trapezoid", n=n)
        simp_val = debye_integral(T, theta_d, method="simpson", n=n)
        diff = abs(trap_val - simp_val)
        print(f"{T:<10.1f} {y:<10.4f} {trap_val:<18.10f} {simp_val:<18.10f} {diff:<15.2e}")
    
    # 误差收敛性分析
    print(f"\n2. 误差收敛性分析 (T=100K):")
    print("-" * 70)
    T_test = 100.0
    ref_n = 10000
    ref_val = debye_integral(T_test, theta_d, method="simpson", n=ref_n)
    print(f"参考值 (Simpson, n={ref_n}): {ref_val:.12f}")
    print("-" * 70)
    print(f"{'n':<10} {'Trapezoid Error':<20} {'Simpson Error':<20} {'Error Ratio':<15}")
    print("-" * 70)
    
    for n in [10, 20, 50, 100, 200]:
        trap_val = debye_integral(T_test, theta_d, method="trapezoid", n=n)
        simp_val = debye_integral(T_test, theta_d, method="simpson", n=n if n % 2 == 0 else n+1)
        trap_err = abs(trap_val - ref_val)
        simp_err = abs(simp_val - ref_val)
        ratio = trap_err / simp_err if simp_err > 0 else float('inf')
        print(f"{n:<10} {trap_err:<20.2e} {simp_err:<20.2e} {ratio:<15.1f}")
    
    print("\n结论:")
    print("- Simpson 法的精度显著高于梯形法（误差比约 10²~10³ 量级）")
    print("- 梯形法误差 ~ O(h²)，Simpson 法误差 ~ O(h⁴)")
    print("- 低温时 y=θD/T 较大，需要更多分段数保证精度")
