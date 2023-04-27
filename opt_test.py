import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

###########################################
# script parameters
def f(x):
    if x > 0:
        return x
    else:
        return -0.1 * x

def f_grad(x):
    if x > 0:
        return 1
    else:
        return -0.1

METHOD_TO_LEARNING_RATE = {
    'Adam': 0.01,
    'GD': 0.00008,
    'rmsprop_with_Nesterov_momentum': 0.008,
    'rmsprop_with_momentum': 0.001,
    'rmsprop': 0.02,
    'momentum': 0.00008,
    'Nesterov': 0.008,
    'Adadelta': None,
    }
X0 = 2
METHOD = 'rmsprop'
# METHOD = 'Adam'
LEARNING_RATE = METHOD_TO_LEARNING_RATE[METHOD]

MOMENTUM_DECAY_FACTOR = 0.9
RMSPROP_SQUARED_GRADS_AVG_DECAY_FACTOR = 0.9
ADADELTA_DECAY_FACTOR = 0.9
RMSPROP_EPSILON = 1e-10
ADADELTA_EPSILON = 1e-6
ADAM_EPSILON = 1e-10
ADAM_SQUARED_GRADS_AVG_DECAY_FACTOR = 0.999
ADAM_GRADS_AVG_DECAY_FACTOR = 0.9

INTERVAL = 9e2
INTERVAL = 1
INTERVAL = 3e2
INTERVAL = 3e1
###########################################

def plot_func(axe, f):
    xs = np.arange(-X0 * 0.5, X0 * 1.05, abs(X0) / 100)
    vf = np.vectorize(f)
    ys = vf(xs)
    return axe.plot(xs, ys, color='grey')

def next_color(color, f):
    color[1] -= 0.01
    if color[1] < 0:
        color[1] = 1
    return color[:]

def update(frame):
    global k, x, prev_step, squared_grads_decaying_avg, \
           squared_prev_steps_decaying_avg, grads_decaying_avg

    if METHOD in ('momentum', 'Nesterov', 'rmsprop_with_momentum',
                  'rmsprop_with_Nesterov_momentum'):
        step_momentum_portion = MOMENTUM_DECAY_FACTOR * prev_step
    if METHOD in ('Nesterov', 'rmsprop_with_Nesterov_momentum'):
        gradient = f_grad(x + step_momentum_portion)
    else:
        gradient = f_grad(x)

    if METHOD == 'GD':
        step = -LEARNING_RATE * gradient
    elif METHOD in ('momentum', 'Nesterov'):
        step = step_momentum_portion - LEARNING_RATE * gradient
    elif METHOD in ('rmsprop', 'rmsprop_with_momentum',
                    'rmsprop_with_Nesterov_momentum'):
        squared_grads_decaying_avg = (
            RMSPROP_SQUARED_GRADS_AVG_DECAY_FACTOR * squared_grads_decaying_avg +
            (1 - RMSPROP_SQUARED_GRADS_AVG_DECAY_FACTOR) * gradient ** 2)
        grads_rms = np.sqrt(squared_grads_decaying_avg + RMSPROP_EPSILON)
        if METHOD == 'rmsprop':
            step = -LEARNING_RATE / grads_rms * gradient
        else:
            assert(METHOD in ('rmsprop_with_momentum',
                              'rmsprop_with_Nesterov_momentum'))
            print(f'LEARNING_RATE / grads_rms * gradient: {LEARNING_RATE / grads_rms * gradient}')
            step = step_momentum_portion - LEARNING_RATE / grads_rms * gradient
    elif METHOD == 'Adadelta':
        gradient = f_grad(x)
        squared_grads_decaying_avg = (
            ADADELTA_DECAY_FACTOR * squared_grads_decaying_avg +
            (1 - ADADELTA_DECAY_FACTOR) * gradient ** 2)
        grads_rms = np.sqrt(squared_grads_decaying_avg + ADADELTA_EPSILON)
        squared_prev_steps_decaying_avg = (
            ADADELTA_DECAY_FACTOR * squared_prev_steps_decaying_avg +
            (1 - ADADELTA_DECAY_FACTOR) * prev_step ** 2)
        prev_steps_rms = np.sqrt(squared_prev_steps_decaying_avg + ADADELTA_EPSILON)
        step = - prev_steps_rms / grads_rms * gradient
    elif METHOD == 'Adam':
        squared_grads_decaying_avg = (
            ADAM_SQUARED_GRADS_AVG_DECAY_FACTOR * squared_grads_decaying_avg +
            (1 - ADAM_SQUARED_GRADS_AVG_DECAY_FACTOR) * gradient ** 2)
        unbiased_squared_grads_decaying_avg = (
            squared_grads_decaying_avg /
            (1 - ADAM_SQUARED_GRADS_AVG_DECAY_FACTOR ** (k + 1)))
        grads_decaying_avg = (
            ADAM_GRADS_AVG_DECAY_FACTOR * grads_decaying_avg +
            (1 - ADAM_GRADS_AVG_DECAY_FACTOR) * gradient)
        unbiased_grads_decaying_avg = (
            grads_decaying_avg /
            (1 - ADAM_GRADS_AVG_DECAY_FACTOR ** (k + 1)))
        step = - (LEARNING_RATE /
                  (np.sqrt(unbiased_squared_grads_decaying_avg) + ADAM_EPSILON) *
                  unbiased_grads_decaying_avg)

    x += step
    prev_step = step
    k += 1

    color = next_color(cur_color, f)

    print(f'k: {k}\n'
          f'x: {x}\n'
          f'step: {step}\n'
          f'gradient: {gradient}\n')

    k_x_marker, = k_and_x.plot(k, x, '.', color=color)
    x_y_marker, = x_and_y.plot(x, f(x), '.', color=color)

    return k_x_marker, x_y_marker

k = 0
x = X0
cur_color = [0, 1, 1]
prev_step = 0
squared_grads_decaying_avg = 0
squared_prev_steps_decaying_avg = 0
grads_decaying_avg = 0

fig, (k_and_x, x_and_y) = plt.subplots(1, 2, figsize=(9,5))
k_and_x.set_xlabel('t')
k_and_x.set_ylabel('x', rotation=0)
x_and_y.set_xlabel('x')
x_and_y.set_ylabel('f(x)', rotation=0)
plot_func(x_and_y, f)
x_and_y.plot(x, f(x), '.', color=cur_color[:])
k_and_x.plot(k, x, '.', color=cur_color[:])
plt.tight_layout()

ani = FuncAnimation(fig, update, frames=300, blit=True, repeat=False, interval=INTERVAL)
ani.save(f'{METHOD}_animation.gif', writer='imagemagick', fps=60)