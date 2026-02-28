import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, RadioButtons


class FourierSeries:
    def __init__(self, func, L, terms=10):
        self.func = func
        self.L = L
        self.terms = terms

    def calculate_a0(self, N=1000):
        x = np.linspace(-self.L, self.L, N)
        y = self.func(x)
        integral = np.trapezoid(y, x)
        a0 = integral/(self.L)
        return a0

    def calculate_an(self, n, N=1000):
        x = np.linspace(-self.L, self.L, N)
        y = self.func(x)*np.cos((n*np.pi*x)/self.L)
        integral = np.trapezoid(y, x)
        an = integral/self.L
        return an

    def calculate_bn(self, n, N=1000):
        x = np.linspace(-self.L, self.L, N)
        y = self.func(x)*np.sin((n*np.pi*x)/self.L)
        integral = np.trapezoid(y, x)
        bn = integral/self.L
        return bn

    def approximate(self, x):
        a0 = self.calculate_a0()
        result = a0/2
        for i in range(1, self.terms+1):
            an = self.calculate_an(i)
            bn = self.calculate_bn(i)
            result += an*np.cos((i*np.pi*x)/self.L)
            result += bn*np.sin((i*np.pi*x)/self.L)
        return result

    def plot(self, ax, wave_type="square"):
        """
        Step 5: Plot the original function and its Fourier series approximation.
        Now plots multiple periods.
        """
        numOfPeriods = 3
        x_range = numOfPeriods * 2 * self.L
        x = np.linspace(-x_range, x_range, 2000)

        # Compute original function values
        original = self.func(x)  # Implement this

        # Compute Fourier series approximation
        approximation = self.approximate(x)  # Implement this

        # Clear axis and Plotting
        ax.clear()
        ax.plot(x, original, label="Original Function",
                color="blue", alpha=0.5)
        ax.plot(x, approximation,
                label=f"Fourier Series (N={self.terms})", color="red", linestyle="--")

        # Dynamic Y-limits to ensure full view is seen for all wave types
        if wave_type == "sawtooth":
            # Sawtooth goes from -pi to +pi
            ax.set_ylim(-3.5, 3.5)
        elif wave_type == "cubic":
            # Cubic x^3 on -1 to 1 ranges from -1 to 1.
            ax.set_ylim(-1.5, 1.5)
        elif wave_type == "pulse":
            ax.set_ylim(-0.5, 1.5)
        else:
            # Square, Triangle are roughly +/- 1
            ax.set_ylim(-1.5, 1.5)

        # Set X-limits to show multiple periods
        if wave_type == "cubic":
            ax.set_xlim(-6, 6)
        else:
            ax.set_xlim(-4 * np.pi, 4 * np.pi)

        ax.legend(loc='upper right')
        ax.grid(True)
        ax.set_title(
            f"Fourier Series Approximation: {wave_type.replace('_', ' ').title()}")


def wrap(x, left, right):
    period = right - left
    return (x-left) % period+left


def target_function(x, function_type="square"):
    """
    Defines target functions.
    """
    if function_type == "square":
        return np.where(np.sin(x) > 0, 1.0, -1.0)

    elif function_type == "sawtooth":
        return wrap(x, -np.pi, np.pi)

    elif function_type == "triangle":
        temp = wrap(x, -np.pi, np.pi)
        return 2*np.abs(temp/np.pi) - 1

    elif function_type == "cubic":
        temp = wrap(x, -1.0, 1.0)
        return temp**3

    elif function_type == "pulse":
        temp = wrap(x, -np.pi, np.pi)
        return np.where(np.abs(temp) < 0.1, 1.0, 0.0)

    else:
        raise ValueError("Invalid function_type.")


def get_half_period(wave_type):
    if (wave_type == "cubic"):
        return 1.0
    else:
        return np.pi


# Example of using these functions in the FourierSeries class with Sliders
if __name__ == "__main__":
    initial_terms = 1  # Start with 1 term
    initial_wave = "square"
    L = get_half_period(initial_wave)  # Half-period for initial function

    # Create the plot figure and axis
    fig_plot, ax_plot = plt.subplots(figsize=(10, 6))

    # Create the widgets figure
    fig_widgets = plt.figure(figsize=(8, 4))
    current_func = (lambda x: target_function(x, initial_wave))
    fs = FourierSeries(current_func, L, initial_terms)
    # Pass the wave type to plot for correct axis scaling
    fs.plot(ax_plot, initial_wave)

    # 1. Radio Buttons (Positioned at the bottom left)
    ax_radio = fig_widgets.add_axes(
        [0.05, 0.1, 0.15, 0.3], facecolor='#f0f0f0')
    radio = RadioButtons(ax_radio, ('square', 'sawtooth',
                         'triangle', 'cubic', 'pulse'))

    # 2. Slider Axes (Positioned above the radio buttons)
    ax_n = fig_widgets.add_axes([0.25, 0.6, 0.6, 0.1])

    # Slider for N (Number of terms)
    slider_n = Slider(
        ax=ax_n,
        label='Harmonics (N)',
        valmin=1,
        valmax=500,
        valinit=initial_terms,
        valstep=1
    )

    # --- Update Logic ---

    def update(val):
        """Callback when slider moves."""
        n = int(slider_n.val)
        wave_type = radio.value_selected

        # Update the FourierSeries object
        fs.terms = n
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)

        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    def change_wave(label):
        """Callback when radio button changes."""
        wave_type = label

        # Update the FourierSeries object
        fs.terms = int(slider_n.val)  # Keep current N
        fs.L = get_half_period(wave_type)
        fs.func = lambda x: target_function(x, wave_type)

        # Re-plot (Pass wave_type to handle axis scaling)
        fs.plot(ax_plot, wave_type)
        fig_plot.canvas.draw_idle()

    # Connect widgets to functions
    slider_n.on_changed(update)
    radio.on_clicked(change_wave)

    plt.show()
