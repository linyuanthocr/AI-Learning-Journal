import numpy as np
import matplotlib.pyplot as plt

# 定义学习率函数生成器
def get_expon_lr_func(
    lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param lr_init: float, initial learning rate
    :param lr_final: float, final learning rate
    :param lr_delay_steps: int, number of steps for delay phase
    :param lr_delay_mult: float, multiplier for initial LR during delay phase
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        # Ensure step is treated as a float for calculations
        step = float(step)

        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0

        # Calculate delay rate
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            # delay_rate starts at lr_delay_mult and decays towards 1.0 over lr_delay_steps
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0

        # Calculate log-linear interpolated learning rate
        # This is equivalent to exponential decay in linear space
        t = np.clip(step / max_steps, 0, 1)

        # Handle potential log(0) or log of small numbers
        # Ensure lr_init and lr_final are positive for log
        log_lr_init = np.log(max(1e-10, lr_init))
        log_lr_final = np.log(max(1e-10, lr_final))

        # Perform log-linear interpolation
        log_lerp_val = log_lr_init * (1 - t) + log_lr_final * t

        # Convert back from log space to linear space
        linear_lerp = np.exp(log_lerp_val)

        # Apply delay rate
        return delay_rate * linear_lerp

    return helper

# --- 设置参数并生成曲线所需数据 ---

# 绘图参数 - 你可以修改这些值来查看不同效果
lr_init = 0.01      # 初始学习率
lr_final = 0.0001   # 最终学习率
max_steps = 1000000 # 最大步数

# 情况 1: 没有延迟衰减 (lr_delay_steps = 0)
# lr_delay_mult = 1.0 在 lr_delay_steps = 0 时没有效果，因为 delay_rate 直接设为 1.0
lr_func_no_delay = get_expon_lr_func(
    lr_init=lr_init,
    lr_final=lr_final,
    lr_delay_steps=0,
    lr_delay_mult=1.0,
    max_steps=max_steps
)

# 情况 2: 包含延迟衰减
delay_steps = int(max_steps * 0.1) # 延迟 10% 的总步数 (例如 100000 步)
# delay_mult < 1.0 会导致开始时学习率较低，然后平滑上升
# delay_mult > 1.0 会导致开始时学习率较高，然后平滑下降
delay_mult = 0.1 # 在延迟阶段，将正常的指数衰减学习率乘以一个从 0.1 平滑过渡到 1.0 的因子

lr_func_with_delay = get_expon_lr_func(
    lr_init=lr_init,
    lr_final=lr_final,
    lr_delay_steps=delay_steps,
    lr_delay_mult=delay_mult,
    max_steps=max_steps
)

# 生成步数序列用于绘图
# 使用更多的点可以使曲线更平滑
steps = np.linspace(0, max_steps, 800) # 生成 800 个步数点

# 计算对应步数下的学习率
lr_no_delay_values = np.array([lr_func_no_delay(step) for step in steps])
lr_with_delay_values = np.array([lr_func_with_delay(step) for step in steps])

# --- 绘制曲线 ---

plt.figure(figsize=(12, 7)) # 调整图表大小

# 绘制没有延迟衰减的曲线
plt.plot(steps, lr_no_delay_values, label='Standard Exponential Decay', color='blue', linewidth=2)

# 绘制有延迟衰减的曲线
plt.plot(steps, lr_with_delay_values, label=f'With Delay (delay_steps={delay_steps}, delay_mult={delay_mult})', color='red', linewidth=2)

# 添加图表元素
plt.xlabel('Optimization Step', fontsize=12)
plt.ylabel('Learning Rate', fontsize=12)
plt.title('Learning Rate Schedule Over Steps', fontsize=14)
plt.legend(fontsize=10)
plt.grid(True, which="both", linestyle='--', alpha=0.7) # 改进网格样式

# 调整坐标轴范围以更好地展示曲线
plt.ylim(bottom=0) # 确保 Y 轴从 0 开始
plt.xlim(0, max_steps) # 确保 X 轴范围正确

# 可以选择将 Y 轴设置为对数尺度，更清晰地看到指数衰减过程（但对延迟阶段可能不友好）
# plt.yscale('log')
# plt.ylabel('Learning Rate (log scale)', fontsize=12)


plt.tight_layout() # 自动调整布局
plt.show() # 显示图表
