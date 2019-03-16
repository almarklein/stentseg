e = 2.718281828459045


def gauss(t, sigma):
    """ Take sample from Gaussian function with the given sigma.
    """
    return e ** (-t ** 2 / (2 * sigma ** 2))


def heavy(t):
    """ Take sample from the heavyside function (a.ka. unit step function).
    """
    return float(t > 0)


def get_motion_pattern(*, A=1, T=1, N=20, top=0.35, offset=0.1, extra=None):
    """ Get a motion pattern based on the given parameters.

    Returns a tuple of (time samples, amplitudes).

    The pattern first rises to a peak and then falls off again. It is
    composed of a series of (partial) Gaussian functions:

        G(t-t_top)_s1 * (1 - H(t-t_top)) + G(t-t_top)_s2 * H(t-t_top)

    where G and H are the Gaussian and heavyside functions, s1 and s2 are
    the sigma's of the respective Gaussian functions, and t_top is the time
    of the peak. The Gaussian is defined as:

        G(t)_s = e ^ (-t^2 / (2 * s^2))

    Parameters (only keyword arguments are allowed):

    * A: the amplitude of the pattern.
    * T: the period of the pattern, in seconds.
    * N: the number of samples (the returned samples do *not* include
      the sample at t=T, which is the same as the value at t=0)
    * top: the temporal location of the peak expressed as a fraction of the T,
      e.g. 0.4 corresponds to 40%.
    * extra: an extra (Gaussian) peak to include. Should be a sequence
      of 3 elements (a, b, c) representing amplitude, peak top and
      sigma: a * G(t-b)_c

    Notes:

    * Both sigma's and peak locations in percentages of T.
    * Two patterns, for which only the T is different have different
      values for the times, but equal amplitude values.

    """

    # Sample tt between 0 and 1
    tt = [i/N for i in range(N)]
    aa = []
    t_top = top # * T

    sigma1 = 0.4 * top
    sigma2 = 0.4 * (1-top)

    # Get amplitudes
    for t in tt:
        a = 0
        a += gauss(t-t_top, sigma1) * (1 - heavy(t-t_top))
        a += gauss(t-t_top, sigma2) * heavy(t-t_top)

        if extra:
            a += extra[0] * gauss(t-extra[1], extra[2])

        aa.append(A * a)

    tt = [t*T for t in tt]
    return tt, aa


def plot_pattern(tt, aa):
    """ Helper function to plot the pattern.
    """
    import visvis as vv

    T = tt[-1] + tt[1]
    amax = max(aa)

    # Repeats, so that you can see whether the signal is continuous
    aa3 = list(aa) * 3
    tt3 = []
    for i in range(0, 3):
        tt3 += [t + i*T for t in tt]

    # Plot the signal and mark a single period
    vv.plot(tt3, aa3, lw=2, lc='b', ms='.', mw=4, mc='b')
    vv.plot([0, 0, T, T, 2*T, 2*T], [0, amax, 0, amax, 0, amax], ls='+', lc='r')


def plot_pattern_plt(tt, aa, label='', ls='--', mark=True, ax=None):
    """ Helper function to plot the pattern.
    """
    import matplotlib.pyplot as plt

    T = tt[-1] + tt[1]
    amax = max(aa)

    # Repeats, so that you can see whether the signal is continuous
    aa3 = list(aa) * 3
    tt3 = []
    for i in range(0, 3):
        tt3 += [t + i*T for t in tt]

    # Plot the signal and mark a single period
    if ax is None:
        ax = plt.gca()
    ax.plot(tt3, aa3, 'bo', mec='b', ls=ls, ms=3, alpha=1, label=label) #bo refers to marker
    if mark==True:
        ax.plot([0, 0], [0, amax], 'b', ls='-', marker = '_')
        ax.plot([T,T], [0, amax], 'b', ls='-', marker = '_')
        ax.plot([2*T, 2*T], [0, amax], 'b', ls='-', marker = '_')



if __name__ == '__main__':
    # Show examples

    import visvis as vv

    vv.figure(1); vv.clf();

    a0=vv.subplot(321); vv.title('default')
    plot_pattern(*get_motion_pattern())

    a1=vv.subplot(322); vv.title('A=2')
    plot_pattern(*get_motion_pattern(A=2))

    a2=vv.subplot(323); vv.title('T=0.6')
    plot_pattern(*get_motion_pattern(T=0.6))

    a3=vv.subplot(324); vv.title('N=10')
    plot_pattern(*get_motion_pattern(N=10))
    a3.axis.showGrid = True

    a4=vv.subplot(325); vv.title('Extra blob')
    plot_pattern(*get_motion_pattern(extra=(0.3, 0.8, 0.03)))

    a5=vv.subplot(326); vv.title('Change more')
    plot_pattern(*get_motion_pattern(A=0.4, T=1.22, N=40, extra=(0.3, 0.8, 0.01)))
