import numpy as np
import matplotlib.pyplot as plt

def ring_potential_point(x: float, y: float, z: float, a: float = 1.0, q: float = 1.0, n_phi: int = 720) -> float:
    """
    计算均匀带电圆环在空间某点产生的电势。
    
    参数:
        x, y, z: 场点坐标
        a: 圆环半径
        q: 电荷参数（总电荷 Q = 4πε₀q，此处q为归一化参数）
        n_phi: 积分离散点数
    
    物理模型:
        圆环位于xy平面，圆心在原点，半径为a。
        圆环上电荷线密度 λ = Q/(2πa) = 4πε₀q/(2πa) = 2ε₀q/a
        电势公式: V = (q/2π) ∫₀²π dφ / √[(x-a·cosφ)² + (y-a·sinφ)² + z²]
    """
    # 生成离散角度
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # 圆环上电荷元的位置 (在xy平面内，z=0)
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    z_ring = 0.0
    
    # 计算场点到各电荷元的距离
    distances = np.sqrt((x - x_ring)**2 + (y - y_ring)**2 + (z - z_ring)**2)
    
    # 数值积分（矩形法/梯形法）
    # 被积函数: 1 / distance
    integrand = 1.0 / distances
    
    # 计算积分并乘以归一化因子 q/(2π)
    potential = (q / (2 * np.pi)) * np.sum(integrand) * d_phi
    
    return float(potential)


def ring_potential_grid(y_grid, z_grid, x0: float = 0.0, a: float = 1.0, q: float = 1.0, n_phi: int = 720):
    """
    计算在 x=x0 平面（通常是yz平面）上的电势分布网格。
    
    参数:
        y_grid: y坐标网格（2D数组，由np.meshgrid生成）
        z_grid: z坐标网格（2D数组）
        x0: x坐标固定值（默认为0，即在yz平面）
        a, q, n_phi: 同ring_potential_point
    
    返回:
        V_grid: 2D电势分布数组，形状与y_grid、z_grid相同
    """
    # 检查输入是否为网格
    if y_grid.ndim != 2 or z_grid.ndim != 2:
        raise ValueError("y_grid and z_grid must be 2D arrays (use np.meshgrid first)")
    
    ny, nz = y_grid.shape
    V_grid = np.zeros((ny, nz))
    
    # 逐点计算电势（向量化计算会更高效，但这里为了清晰使用循环）
    for i in range(ny):
        for j in range(nz):
            y = y_grid[i, j]
            z = z_grid[i, j]
            V_grid[i, j] = ring_potential_point(x0, y, z, a, q, n_phi)
    
    return V_grid


def axis_potential_analytic(z: float, a: float = 1.0, q: float = 1.0) -> float:
    """
    圆环轴线上（x=y=0）电势的解析解，用于验证数值计算。
    
    解析公式: V(0,0,z) = q / sqrt(a² + z²)
    """
    return q / np.sqrt(a * a + z * z)


def compute_electric_field(V_grid, dy, dz):
    """
    通过电势计算电场强度（E = -∇V）。
    
    参数:
        V_grid: 电势网格（2D数组）
        dy, dz: y和z方向的网格间距
    
    返回:
        Ey, Ez: 电场y和z分量（2D数组）
    """
    # 使用numpy的gradient计算梯度（中心差分）
    # np.gradient返回[∂V/∂y, ∂V/∂z]（如果第一个轴是y方向）
    dV_dy, dV_dz = np.gradient(V_grid, dy, dz)
    
    # 电场 E = -∇V
    Ey = -dV_dy
    Ez = -dV_dz
    
    return Ey, Ez


def plot_ring_potential_field(a=1.0, q=1.0, x0=0.0):
    """
    绘制yz平面上的电势等势线和电场矢量分布。
    """
    # 创建坐标网格
    # 避开圆环所在位置附近（y=a, z=0）的奇点，稍微扩大范围
    y_range = np.linspace(-2.5*a, 2.5*a, 100)
    z_range = np.linspace(-2.5*a, 2.5*a, 100)
    Y, Z = np.meshgrid(y_range, z_range)
    
    # 计算电势分布
    print("Computing potential grid...")
    V = ring_potential_grid(Y, Z, x0=x0, a=a, q=q, n_phi=720)
    
    # 计算电场（用于矢量图）
    dy = y_range[1] - y_range[0]
    dz = z_range[1] - z_range[0]
    Ey, Ez = compute_electric_field(V, dy, dz)
    
    # 创建图形
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))
    
    # 绘制等势线（contour）
    # 使用对数级别的等势线，因为电势随距离衰减
    levels = np.logspace(np.log10(V.max()*0.01), np.log10(V.max()), 20)
    contour = ax.contour(Y, Z, V, levels=levels, colors='black', alpha=0.5, linewidths=0.8)
    ax.clabel(contour, inline=True, fontsize=8, fmt='%.3f')  # 标注等势线数值
    
    # 绘制填充等势线（contourf）
    contourf = ax.contourf(Y, Z, V, levels=50, cmap='RdYlBu_r', alpha=0.8)
    cbar = plt.colorbar(contourf, ax=ax, label='Potential V')
    
    # 绘制电场矢量（quiver），每隔一定间隔画一个箭头避免太密
    skip = 8  # 每隔8个点画一个箭头
    ax.quiver(Y[::skip, ::skip], Z[::skip, ::skip], 
              Ey[::skip, ::skip], Ez[::skip, ::skip],
              alpha=0.6, color='white', scale=20, width=0.003)
    
    # 标记圆环位置（在yz平面上，圆环投影为两个点 (y=±a, z=0)）
    ax.plot(a, 0, 'ko', markersize=8, label='Ring location (y=±a)')
    ax.plot(-a, 0, 'ko', markersize=8)
    
    # 如果x0=0（在yz平面内），标记圆环的投影线
    circle = plt.Circle((0, 0), a, fill=False, color='red', linestyle='--', 
                        linewidth=2, alpha=0.5, label='Ring projection')
    ax.add_patch(circle)
    
    ax.set_xlabel('y')
    ax.set_ylabel('z')
    ax.set_title(f'Electric Potential and Field of a Charged Ring\n'
                 f'(x={x0}, a={a}, q={q})')
    ax.legend()
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('ring_potential_field.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    return fig, ax


# 验证函数：与轴线解析解对比
def verify_calculation():
    """验证数值计算与轴线解析解的一致性"""
    a, q = 1.0, 1.0
    z_points = np.linspace(0.1, 3.0, 20)  # 避开z=0奇点
    
    V_numerical = [ring_potential_point(0, 0, z, a, q, n_phi=1000) for z in z_points]
    V_analytic = [axis_potential_analytic(z, a, q) for z in z_points]
    
    error = np.abs(np.array(V_numerical) - np.array(V_analytic))
    print(f"Max error vs analytic: {error.max():.6e}")
    
    return np.all(error < 1e-4)


if __name__ == "__main__":
    # 1. 验证数值计算正确性
    print("Verifying numerical calculation against analytic solution...")
    is_accurate = verify_calculation()
    print(f"Verification passed: {is_accurate}")
    
    # 2. 绘制电势和电场分布图（yz平面，x=0）
    print("\nGenerating visualization...")
    plot_ring_potential_field(a=1.0, q=1.0, x0=0.0)
    
    # 3. 测试单点计算
    print("\nTest single point calculation:")
    V_test = ring_potential_point(0, 0, 1.0, a=1.0, q=1.0)
    V_analytic_test = axis_potential_analytic(1.0, a=1.0, q=1.0)
    print(f"Numerical V(0,0,1) = {V_test:.6f}")
    print(f"Analytic V(0,0,1)  = {V_analytic_test:.6f}")
