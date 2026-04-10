import numpy as np
from numpy.polynomial.legendre import leggauss

# 引力常数 (m^3 kg^-1 s^-2)
G = 6.674e-11


def gauss_legendre_2d(func, ax: float, bx: float, ay: float, by: float, n: int = 40) -> float:
    """
    使用二维高斯-勒让德积分计算双重积分 ∬ func(x,y) dx dy。
    
    参数:
        func: 被积函数，接受两个参数 func(x, y)，返回标量
        ax, bx: x 方向的积分区间 [ax, bx]
        ay, by: y 方向的积分区间 [ay, by]
        n: 每个维度上的高斯-勒让德积分点数（默认 40）
    
    返回:
        双重积分的数值结果
    
    原理:
        在标准区间 [-1, 1] 上，高斯-勒让德积分点为 t_i，权重为 w_i。
        通过变量代换将 [ax, bx] 和 [ay, by] 映射到 [-1, 1]：
            x = (bx-ax)/2 * t + (bx+ax)/2
            y = (by-ay)/2 * t + (by+ay)/2
        积分结果 = ((bx-ax)/2) * ((by-ay)/2) * Σ_i Σ_j w_i * w_j * func(x_i, y_j)
    """
    # 获取高斯-勒让德积分点和权重（在 [-1, 1] 区间）
    t, w = leggauss(n)
    
    # 将积分点从 [-1, 1] 映射到 [ax, bx] 和 [ay, by]
    # x 坐标
    x_nodes = (bx - ax) / 2 * t + (bx + ax) / 2
    # y 坐标  
    y_nodes = (by - ay) / 2 * t + (by + ay) / 2
    
    # 计算雅可比行列式因子（区间变换的缩放系数）
    jacobian_x = (bx - ax) / 2
    jacobian_y = (by - ay) / 2
    
    # 计算双重积分
    # 方法：对 x 和 y 的积分点进行网格化，计算所有点的函数值，
    # 然后用外积权重求和
    integral = 0.0
    
    # 使用向量化计算提高效率
    # 创建网格：xx[i,j] = x_nodes[i], yy[i,j] = y_nodes[j]
    xx, yy = np.meshgrid(x_nodes, y_nodes, indexing='ij')
    
    # 计算所有网格点上的函数值
    f_values = func(xx, yy)
    
    # 计算权重的外积：w_x[i] * w_y[j]
    weights = np.outer(w, w)  # shape: (n, n)
    
    # 加权求和
    integral = np.sum(weights * f_values) * jacobian_x * jacobian_y
    
    return float(integral)


def plate_force_z(z: float, L: float = 10.0, M_plate: float = 1.0e4, 
                  m_particle: float = 1.0, n: int = 40) -> float:
    """
    计算正方形金属板中心正上方 z 位置处质点受到的引力 F_z。
    
    参数:
        z: 质点距离金属板中心的高度（垂直距离，单位：m）
        L: 金属板边长（默认 10.0 m）
        M_plate: 金属板总质量（默认 10^4 kg = 10 吨）
        m_particle: 测试质点质量（默认 1.0 kg）
        n: 积分点数（默认 40）
    
    返回:
        引力 F_z 的 z 分量（单位：牛顿 N），方向向下（指向金属板）
    
    物理公式:
        F_z = G * σ * m_particle * z * ∬_{-L/2}^{L/2} dx dy / (x² + y² + z²)^(3/2)
        其中 σ = M_plate / L² 是面密度。
    
    注意:
        由于对称性，水平方向 Fx = Fy = 0，只有 z 分量。
        力的方向为负 z 方向（吸引力），但此处返回正值表示大小
        （严格来说 F_z 应为负值，表示指向 -z 方向）。
    """
    # 计算面密度（质量 per 面积）
    sigma = M_plate / (L * L)  # kg/m²
    
    # 定义被积函数：f(x, y) = z / (x² + y² + z²)^(3/2)
    # 注意：由于 z 是常数参数，我们将其闭包在函数内
    def integrand(x, y):
        r_cubed = (x**2 + y**2 + z**2)**(1.5)  # (x² + y² + z²)^(3/2)
        return z / r_cubed
    
    # 使用高斯-勒让德积分计算双重积分
    # 积分区域：x ∈ [-L/2, L/2], y ∈ [-L/2, L/2]
    half_L = L / 2
    integral_result = gauss_legendre_2d(
        integrand, 
        ax=-half_L, bx=half_L, 
        ay=-half_L, by=half_L, 
        n=n
    )
    
    # 计算总引力
    # F_z = G * σ * m_particle * integral_result
    # 注意：引力方向指向金属板（负 z 方向），但题目可能只需要大小
    # 这里返回实际的 z 分量（应为负值），但物理意义上我们通常关心大小
    force_z = G * sigma * m_particle * integral_result
    
    # 根据物理，力应该指向金属板（-z 方向），所以如果返回负值是正确的
    # 但如果只需要大小，可以取 abs。这里保留符号以示物理正确性。
    return force_z


def force_curve(z_values, L: float = 10.0, M_plate: float = 1.0e4, 
                m_particle: float = 1.0, n: int = 40):
    """
    计算一系列 z 值对应的引力 F_z 数组。
    
    参数:
        z_values: z 坐标的一维数组或列表（单位：m）
        L, M_plate, m_particle, n: 同 plate_force_z
    
    返回:
        与 z_values 对应的 F_z 数组（单位：N）
    """
    # 转换为 numpy 数组以便迭代
    z_array = np.asarray(z_values)
    
    # 计算每个 z 对应的力
    forces = np.array([
        plate_force_z(z, L, M_plate, m_particle, n) 
        for z in z_array
    ])
    
    return forces


def verify_analytical_limit():
    """
    验证数值计算的渐近行为：当 z >> L 时，方板应该近似为点质量。
    此时 F_z ≈ G * M_plate * m_particle / z²
    """
    L = 10.0
    M = 1.0e4
    m = 1.0
    
    # 测试大的 z 值
    z_test = np.array([100.0, 200.0, 500.0])  # 远大于 L=10
    
    print("Verification (z >> L limit):")
    print(f"Plate mass M = {M} kg, size L = {L} m")
    print("-" * 60)
    
    for z in z_test:
        F_numerical = plate_force_z(z, L, M, m, n=50)
        # 点质量近似：F = G * M * m / z^2（注意方向）
        F_point_mass = G * M * m / (z**2)
        
        # 数值结果应该略小于点质量近似（因为板有扩展，距离效率降低）
        error = abs(abs(F_numerical) - F_point_mass) / F_point_mass * 100
        
        print(f"z = {z:6.1f} m: F_numerical = {abs(F_numerical):.6e} N, "
              f"F_point_mass = {F_point_mass:.6e} N, "
              f"diff = {error:.2f}%")


def plot_force_curve():
    """
    绘制 z ∈ [0.2, 10] 范围内的引力曲线，用于可视化。
    """
    import matplotlib.pyplot as plt
    
    # 生成 z 值（对数均匀分布更能展示变化）
    z_values = np.linspace(0.2, 10.0, 100)
    
    # 计算力（取绝对值用于绘图，因为力是吸引力，z 分量为负）
    forces = np.abs(force_curve(z_values, L=10.0, M_plate=1.0e4, m_particle=1.0, n=50))
    
    plt.figure(figsize=(10, 6))
    
    # 绘制力-距离曲线
    plt.subplot(1, 2, 1)
    plt.plot(z_values, forces, 'b-', linewidth=2, label='|F_z| (numerical)')
    plt.xlabel('z (m)')
    plt.ylabel('|F_z| (N)')
    plt.title('Gravitational Force vs Distance')
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # 绘制对数坐标以显示幂律行为
    plt.subplot(1, 2, 2)
    plt.loglog(z_values, forces, 'r-', linewidth=2, label='|F_z|')
    plt.xlabel('z (m)')
    plt.ylabel('|F_z| (N)')
    plt.title('Log-Log Plot of Force vs Distance')
    plt.grid(True, alpha=0.3, which='both')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('plate_gravity_curve.png', dpi=150)
    plt.show()
    
    # 打印几个关键数据点
    print("\nSample data points:")
    sample_z = [0.2, 0.5, 1.0, 2.0, 5.0, 10.0]
    sample_f = np.abs(force_curve(sample_z))
    for z, f in zip(sample_z, sample_f):
        print(f"z = {z:4.1f} m: |F_z| = {f:.6e} N")


if __name__ == "__main__":
    # 任务 3：对 z ∈ [0.2, 10] 进行计算并展示结果
    print("Task 3: Computing gravitational force for z ∈ [0.2, 10] meters")
    print("=" * 70)
    
    # 创建 z 值数组（100个点，包含 0.2 和 10）
    z_range = np.linspace(0.2, 10.0, 100)
    
    # 计算力曲线
    forces = force_curve(z_range, L=10.0, M_plate=1.0e4, m_particle=1.0, n=40)
    
    # 打印结果摘要
    print(f"Computed {len(z_range)} points from z=0.2m to z=10.0m")
    print(f"Force at z=0.2m: {forces[0]:.6e} N")
    print(f"Force at z=10m:  {forces[-1]:.6e} N")
    print(f"Force ratio F(0.2)/F(10): {abs(forces[0]/forces[-1]):.2f}")
    
    # 验证远场近似
    print("\n" + "=" * 70)
    verify_analytical_limit()
    
    # 绘制曲线（如果安装了 matplotlib）
    try:
        print("\n" + "=" * 70)
        print("Generating plot...")
        plot_force_curve()
    except Exception as e:
        print(f"Plot generation skipped: {e}")
