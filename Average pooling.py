import logging

# 配置日志记录
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def distribute_value(dZ, shape):
    """
    分配值函数

    参数:
    dZ (float): 输入值
    shape (tuple): 形状参数

    返回:
    float: 分配后的值
    """
    # 验证输入参数
    if not isinstance(dZ, (int, float)):
        raise ValueError("dZ 必须是数值类型")
    if not isinstance(shape, tuple) or len(shape) != 2:
        raise ValueError("shape 必须是长度为2的元组")
    if not all(isinstance(dim, int) and dim > 0 for dim in shape):
        raise ValueError("shape 的元素必须是正整数")

    # 计算分配值
    try:
        distributed_value = dZ / (shape[0] * shape[1])
    except ZeroDivisionError:
        raise ValueError("shape 的元素不能为零")

    # 记录日志
    logging.info(f'distributed value = {distributed_value}')

    return distributed_value

# 示例调用
if __name__ == "__main__":
    result = distribute_value(10, (2, 2))
    print(result)
