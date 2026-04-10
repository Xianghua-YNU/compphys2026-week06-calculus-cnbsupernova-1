import numpy as np

def ring_potential_point(x, y, z, a=1.0, q=1.0, n_phi=720):
    """
    计算均匀带电圆环在空间某点产生的电势（数值积分）。
    """
    # 生成离散角度
    phi = np.linspace(0, 2 * np.pi, n_phi, endpoint=False)
    d_phi = 2 * np.pi / n_phi
    
    # 圆环上电荷元的位置 (在xy平面，z=0)
    x_ring = a * np.cos(phi)
    y_ring = a * np.sin(phi)
    
    # 计算场点到各电荷元的距离
    distances = np.sqrt((x - x_ring)**2 + (y - y_ring)**2 + z**2)
    
    # 数值积分（矩形法）
    integrand = 1.0 / distances
    potential = (q / (2 * np.pi)) * np.sum(integrand) * d_phi
    
    return float(potential)


def ring_potential_grid(y_grid, z_grid, x0=0.0, a=1.0, q=1.0, n_phi=720):
    """
    计算在 x=x0 平面上的电势分布网格。
    接受 1D 数组（自动创建meshgrid）或 2D 数组输入。
    """
    y_arr = np.asarray(y_grid)
    z_arr = np.asarray(z_grid)
    
    # 如果输入是1D数组，创建2D网格
    if y_arr.ndim == 1 and z_arr.ndim == 1:
        Y, Z = np.meshgrid(y_arr, z_arr, indexing='ij')
        ny, nz = Y.shape
    elif y_arr.ndim == 2 and z_arr.ndim == 2:
        Y, Z = y_arr, z_arr
        ny, nz = Y.shape
    else:
        # 处理其他情况
        Y, Z = np.broadcast_arrays(y_arr, z_arr)
        ny, nz = Y.shape
    
    # 计算每个网格点的电势
    V_grid = np.zeros((ny, nz))
    for i in range(ny):
        for j in range(nz):
            V_grid[i, j] = ring_potential_point(x0, Y[i, j], Z[i, j], a, q, n_phi)
    
    return V_grid


def axis_potential_analytic(z, a=1.0, q=1.0):
    """
    圆环轴线上（x=y=0）电势的解析解。
    
    解析公式: V(0,0,z) = q / sqrt(a^2 + z^2)
    
    这是用于验证数值计算正确性的参考解。
    """
    return q / np.sqrt(a * a + z * z)
