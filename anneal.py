"""
Create annealing schedule for learning rate / regularization lambda.
"""
import matplotlib.pyplot as plt
import numpy as np


class AnnealingSchedule:
    def __init__(self, n_steps, n_steps_mid, sharpness, initial_value=0.0, final_value=1.0):
        """
        Ramp up lienarly or suddenly or gradually (with the sharpness parameter).
        :param n_steps:  Number of steps in the schedule.
        :param n_steps_mid:  Which step number is to be half way from initial value to the final value.
        :param sharpness: 
        :param initial_value: The initial value of the schedule.
        :param final_value: The final value of the schedule.
        """
        self.start_val = initial_value
        self.final_val = final_value
        self.n_steps = n_steps
        self.n_steps_mid = n_steps_mid
        self.sharpness = sharpness

def make_annealing_schedule(inflection_point=0.2, sharpness=10.0, init_value=0.0, final_value=1.0, num_steps=100):
    """
    Create a linear annealing schedule.
    """
    if sharpness>0:
        t = np.linspace(0.0, 1.0, num_steps)
        v = np.tanh((t - inflection_point) * sharpness)
        v = (v - v.min()) / (v.max() - v.min())
        return v * (final_value - init_value) + init_value
    return np.linspace(init_value, final_value, num_steps)


def test_annealing_sched():
    inflection_points = 0.1, 0.25, 0.5, 0.75
    sharpnesses = [0,1,5,10,20,50,100]
    n_steps = 1000
    fig, ax = plt.subplots(nrows=len(inflection_points), ncols=len(sharpnesses), figsize=(15, 8))
    for j, inflection_point in enumerate(inflection_points):
        for i, sharpness in enumerate(sharpnesses):
            schedule = make_annealing_schedule(inflection_point=inflection_point, sharpness=sharpness,
                                                  num_steps=n_steps)

            ax[j, i].plot(schedule)
            title = f"inflection: {inflection_point},\nsharpness: {sharpness:.2f}"
            ax[j, i].set_title(title, fontsize=8)
            # turn off axes
            ax[j, i].axis('off')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    test_annealing_sched()