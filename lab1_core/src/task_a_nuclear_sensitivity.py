import numpy as np


def rate_3alpha(T: float) -> float:
    """
    计算3-α反应率 q(T)。
    
    公式：q(T) = 5.09×10^11 × T8^(-3) × exp(-44.027/T8)
    其中 T8 = T / 10^8
    
    参数:
        T: 开尔文温度 (K)
    返回:
        q(T): 反应率 (温度相关部分)
    """
    T8 = T / 1.0e8
    # 防止非正温度导致数值错误
    if T8 <= 0:
        return 0.0
    return 5.09e11 * (T8 ** (-3.0)) * np.exp(-44.027 / T8)


def finite_diff_dq_dT(T0: float, h: float = 1e-8) -> float:
    """
    使用前向差分近似计算 dq/dT 在 T0 处的值。
    
    ΔT = h * T0 （注意：h是相对步长，绝对增量是 h*T0）
    dq/dT ≈ [q(T0 + ΔT) - q(T0)] / ΔT
    
    参数:
        T0: 参考温度 (K)
        h: 相对步长（默认 1e-8）
    返回:
        dq/dT 在 T0 处的近似值
    """
    if T0 <= 0:
        raise ValueError("Temperature T0 must be positive")
    
    delta_T = h * T0  # 关键：ΔT = h * T0，不是 h 本身
    q_T0 = rate_3alpha(T0)
    q_T0_plus = rate_3alpha(T0 + delta_T)
    
    return (q_T0_plus - q_T0) / delta_T


def sensitivity_nu(T0: float, h: float = 1e-8) -> float:
    """
    计算温度敏感性指数 ν(T0)。
    
    定义：ν = (d log q / d log T)|_{T0} = (T/q * dq/dT)|_{T0}
    
    物理意义：温度发生相对变化时，反应率会被放大多少倍。
    
    参数:
        T0: 参考温度 (K)
        h: 相对步长（默认 1e-8）
    返回:
        ν(T0): 温度敏感性指数
    """
    if T0 <= 0:
        raise ValueError("Temperature T0 must be positive")
    
    q_T0 = rate_3alpha(T0)
    if q_T0 == 0:
        raise ValueError("q(T0) is zero, cannot compute nu")
    
    dq_dT = finite_diff_dq_dT(T0, h)
    
    # ν = (T0 / q(T0)) * (dq/dT)
    # 注意：必须使用 q(T0)，不是 q(T0+ΔT)
    return (T0 / q_T0) * dq_dT


def nu_table(T_values, h: float = 1e-8):
    """
    计算一系列温度点的敏感性指数列表。
    
    参数:
        T_values: 温度列表或数组 (K)
        h: 相对步长（默认 1e-8）
    返回:
        [(T, nu(T)), ...]: 温度-敏感性指数对的列表
    """
    results = []
    for T in T_values:
        nu = sensitivity_nu(T, h)
        results.append((T, nu))
    return results


# 主程序：计算必算温度点并输出
if __name__ == "__main__":
    # 必算温度点
    T_required = [1.0e8, 2.5e8, 5.0e8, 1.0e9, 2.5e9, 5.0e9]
    
    print("3-α反应率温度敏感性指数 ν 计算结果")
    print("=" * 40)
    
    for T in T_required:
        nu = sensitivity_nu(T)
        print(f"{T:.3e} K : nu = {nu:.2f}")
    
    print("=" * 40)
