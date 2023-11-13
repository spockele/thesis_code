import pandas as pd
import numpy as np
import scipy.signal as spsig
import matplotlib.pyplot as plt
from matplotlib.colors import TABLEAU_COLORS
from case_mgmt import CaseLoader
import helper_functions as hf
import propagation_model as pm


colors = list(TABLEAU_COLORS.keys())


def window_plot():
    """
    Plot of the extreme overlap used in the reconstruction
    """
    n_base = 512
    overlap = 16
    n_fft = n_base * overlap
    y = np.zeros(4 * overlap * n_base)

    x_window = np.arange(0, n_fft).astype(int)
    window = spsig.windows.hann(n_fft)

    plt.figure(1, figsize=(7, 3.5))
    plt.plot(x_window, window, '0.5', label=f'Hanning Windows (N={n_base * overlap})')
    while x_window[-1] <= y.size:
        y[x_window] += window
        plt.plot(x_window, window, '0.5')
        x_window += n_base

    plt.plot(y / (overlap / 2), 'r', label='Corrected overlap-addition')
    plt.plot(y, 'k', label='Uncorrected overlap-addition')

    plt.ylabel('$W(i)$')
    plt.xlabel('$i$')
    plt.legend()
    plt.grid()

    plt.tight_layout()
    plt.show()


def case_plot(project, case):
    """
    Plots of the case receiver positions around the turbine
    :param project: name of the project directory
    :param case: name of the case inside the project directory
    """
    caseloader = CaseLoader(project, case)
    plt.figure(1, )
    ax = plt.subplot()
    ax.set_aspect('equal')

    hub_pos: hf.Cartesian = caseloader.conditions_dict['hub_pos']

    recs = pd.DataFrame(0., columns=['phi'], index=list(caseloader.receiver_dict.keys()))

    for rec_idx, receiver in caseloader.receiver_dict.items():
        x, y, _ = receiver.vec

        recs.loc[rec_idx, :] = receiver.to_spherical(hf.Cartesian(0, 0, 0)).vec[1]

        ax.scatter(-x, y, color='k')
        ax.annotate(f'Mic. {rec_idx + 1}', (-x + 1, y + 1))

    x, y, _ = hub_pos.vec
    r = caseloader.conditions_dict['rotor_radius']
    ax.plot(-x, 0, 'k8')
    ax.plot((-x - r, -x + r), (y, y), 'k')

    ax.grid()
    ax.set_xlabel('-x (m)')
    ax.set_ylabel('y (m)')
    ax.set_xlim(-50, 50)
    ax.set_ylim(-50, 50)
    plt.show()


def ground_effect_plot_arntzen():
    """
    Plot replicating Figure 3.7 from Arntzen (2014, Fig. 3.7)
    - Arntzen, M. (2014). Aircraft noise calculation and synthesis in a non-standard atmosphere [Doctoral thesis,
        Delft University of Technology]. https://doi.org/10.4233/uuid:c56e213c-82db-423d-a5bd-503554653413
    """
    h_s = 100
    h_m = 1.2
    x = np.linspace(1, 2500, 25000)

    r1_arr = np.sqrt((x ** 2) + ((h_s - h_m) ** 2))
    r2_arr = np.sqrt((x ** 2) + ((h_s + h_m) ** 2))

    th = np.arctan2(h_s + h_m, x)

    f = np.array([500.])
    k = 2 * np.pi * f / hf.c

    arntzen_data = pd.read_csv('helper_functions/data/Arntzen_2014_fig3-7_data.csv')

    plt.figure(1, figsize=(6.4, 3.6))

    for si, (surface, sigma_e) in enumerate({'Snow': 29., 'Grass': 200., 'Dirt': 550., 'Asphalt': 10000.}.items()):
        z = pm.ground_impedance(f, sigma_e * 1000)
        ag = np.zeros(r2_arr.shape)
        for ri, r2 in enumerate(r2_arr):
            rp = (z * np.sin(th[ri]) - 1) / (z * np.sin(th[ri]) + 1)
            r1 = r1_arr[ri]
            w = pm.numerical_distance(f, r2, th[ri], z)
            fs = pm.spherical_wave_correction(w)
            q = rp + (1 - rp) * fs
            ag[ri] = 10 * np.log10(1 + ((r1 / r2 * np.abs(q)) ** 2) + (2 * (r1 / r2) * np.abs(q) * np.cos(k * (r2 - r1) + np.angle(q))))[0]

        plt.plot(x, ag, label=surface, color=colors[si])

        plt.plot(arntzen_data.loc[:, f'{surface} x'], arntzen_data.loc[:, f'{surface} y'], ':', color=colors[si])

    plt.xlim(0, 2500)
    plt.ylim(-30, 10)
    plt.xlabel('Distance (m)')
    plt.ylabel('Att (dB)')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()


def ground_effect_plot_embleton():
    """
    Plots replicating Figures 2 and 3 from Embleton et al. (1983, Figs. 2-3).
    - Embleton, T. F. W., Piercy, J. E., & Daigle, G. A. (1983). Effective flow resistivity of ground surfaces
        determined by acoustical measurements. The Journal of the Acoustical Society of America, 74(4), 1239–1244.
        https://doi.org/10.1121/1.390029
    """
    f = np.linspace(1, 10e3, 10000)
    k = 2 * np.pi * f / hf.c

    sigma_e_arr = [10, 32, 100, 320, 1000, 3200, 10000, 32000]
    h_s = 0.31

    plt.figure(1, figsize=(5.5, 4))
    embleton_data = pd.read_csv('helper_functions/data/embleton_1983_fig2_data.csv', )
    x = 15.2
    h_m = 1.22
    r1 = np.sqrt((x ** 2) + ((h_s - h_m) ** 2))
    r2 = np.sqrt((x ** 2) + ((h_s + h_m) ** 2))
    th = np.arctan2(h_s + h_m, x)

    for sei, sigma_e in enumerate(sigma_e_arr):
        z = pm.ground_impedance(f, sigma_e * 1000)
        rp = (z * np.sin(th) - 1) / (z * np.sin(th) + 1)
        w = pm.numerical_distance(f, r2, th, z)
        fs = pm.spherical_wave_correction(w)
        q = rp + (1 - rp) * fs
        ag = 10 * np.log10(1 + ((r1 / r2 * np.abs(q)) ** 2) + (2 * (r1 / r2) * np.abs(q) * np.cos(k * (r2 - r1) + np.angle(q))))

        plt.semilogx(f / 1e3, ag, label=f'$\\sigma_e = {sigma_e}$', color=colors[sei])
        plt.scatter(embleton_data.loc[:, f's{int(sigma_e)}x'], embleton_data.loc[:, f's{int(sigma_e)}y'],
                    marker='.', color=colors[sei])

    plt.scatter([1e-10, ], [1e-10, ], color='k', marker='.', label='Embleton, 1983')

    plt.xlim(.1, 10)
    plt.ylim(-25, 10)
    plt.legend(ncols=3)
    plt.xlabel('$f$ (kHz)')
    plt.ylabel('Att (dB)')
    plt.grid(which='both')
    plt.tight_layout()

    plt.figure(2, figsize=(5.5, 4))
    embleton_data = pd.read_csv('helper_functions/data/embleton_1983_fig3_data.csv', )
    x = 7.62
    h_m = .46
    r1 = np.sqrt((x ** 2) + ((h_s - h_m) ** 2))
    r2 = np.sqrt((x ** 2) + ((h_s + h_m) ** 2))
    th = np.arctan2(h_s + h_m, x)

    for sei, sigma_e in enumerate(sigma_e_arr):
        z = pm.ground_impedance(f, sigma_e * 1000)
        rp = (z * np.sin(th) - 1) / (z * np.sin(th) + 1)
        w = pm.numerical_distance(f, r2, th, z)
        fs = pm.spherical_wave_correction(w)
        q = rp + (1 - rp) * fs
        ag = 10 * np.log10(
            1 + ((r1 / r2 * np.abs(q)) ** 2) + (2 * (r1 / r2) * np.abs(q) * np.cos(k * (r2 - r1) + np.angle(q))))

        plt.semilogx(f / 1e3, ag, label=f'$\\sigma_e = {sigma_e}$', color=colors[sei])
        plt.scatter(embleton_data.loc[:, f's{int(sigma_e)}x'], embleton_data.loc[:, f's{int(sigma_e)}y'],
                    marker='.', color=colors[sei])

    plt.scatter([1e-10, ], [1e-10, ], color='k', marker='.', label='Embleton, 1983')
    plt.xlim(.1, 10)
    plt.ylim(-25, 10)
    plt.legend(ncols=3)
    plt.xlabel('$f$ (kHz)')
    # plt.ylabel('Attenuation (dB)')
    plt.grid(which='both')
    plt.tight_layout()

    plt.show()


def psat_plot():
    """
    Plots equations for the water vapour saturation pressure (Eqs. 1, 2; Bass, 1995)(Eq. 10, Bolton, 1980).
     - Bass, H. E., Sutherland, L. C., Zuckerwar, A. J., Blackstock, D. T., & Hester, D. M. (1995).
        Atmospheric absorption of sound: Further developments. The Journal of the Acoustical Society of America, 97(1),
        680–683. doi: 10.1121/1.412989
     - Bolton, D. (1980). The Computation of Equivalent Potential Temperature. Monthly Weather Review, 108(7),
        1046–1053. doi: 10.1175/1520-0493(1980)108<1046:TCOEPT>2.0.CO;2
    """
    temperature = np.linspace(hf.t_0 - 35, hf.t_0 + 35)
    t_01 = 273.16
    psat_bass_0 = 101325 * 10 ** (10.79586 * (1 - (t_01 / temperature)) - 5.02808 * np.log10(temperature / t_01) +
                                  1.50474e-4 * (1 - 10 ** (-8.29692 * ((temperature / t_01) - 1))) -
                                  4.2873e-4 * (1 - 10 ** (-4.76955 * ((temperature / t_01) - 1))) - 2.2195983
                                  )
    psat_bass_1 = 101325 * 10 ** (-6.8346 * (t_01 / temperature) ** 1.261 + 4.6151)
    psat_bolton = 622.2 * np.exp(17.67 * (temperature - 273.15) / (temperature - 29.65))

    plt.plot(temperature - hf.t_0, psat_bass_0, label='Bass (1995) Eq. 1')
    plt.plot(temperature - hf.t_0, psat_bass_1, label='Bass (1995) Eq. 2')
    plt.plot(temperature - hf.t_0, psat_bolton, label='Bolton (1980) Eq. 10')
    plt.xlim(-35, 35)
    plt.xlabel('$T$ (K)')
    plt.ylabel('$p_{sat}$ (Pa)')
    plt.grid()
    plt.legend()
    plt.tight_layout()
    plt.show()


def atm_absorption_plot():
    """
    Replication of Figure 1 by Bass et al. (1995, Fig. 1) with the papers data and the current implementation
    - Bass, H. E., Sutherland, L. C., Zuckerwar, A. J., Blackstock, D. T., & Hester, D. M. (1995).
        Atmospheric absorption of sound: Further developments. The Journal of the Acoustical Society of America, 97(1),
        680–683. doi: 10.1121/1.412989
    """
    f = np.arange(1e1, 1e6, 1)
    humidities = np.arange(0, 100 + 10, 10)
    temperature = hf.t_0 + 20
    pressure = 101325.

    bass_data = pd.read_csv('helper_functions/data/bass_1995_fig1_data.csv')

    plt.figure(1, )
    for hi, humidity in enumerate(humidities):
        alpha = pm.atm_absorption_coefficient(f, humidity, pressure, temperature)
        plt.loglog(f, 20 * 100 * alpha / np.log(10), label=f'{str(humidities[hi]).zfill(2)} %', color=colors[hi % len(colors)])
        plt.scatter(bass_data.loc[:, f'h{str(humidity).zfill(2)}x'], bass_data.loc[:, f'h{str(humidity).zfill(2)}y'],
                    marker='.', color=colors[hi % len(colors)])

    plt.scatter([1e-10, ], [1e-10, ], color='k', marker='.', label='Bass, 1995')
    plt.xlim(1e1, 1e6)
    plt.ylim(1e-3, 1e4)
    plt.xlabel(f'$f$ (Hz)')
    plt.ylabel('$\\bar{\\alpha}$ (dB) ($\\Delta x = 100$ m)')
    plt.tight_layout()
    plt.legend(ncols=2)
    plt.grid()
    plt.show()


def tool_overview_plots():
    """
    3D Plots to show how the source model works
    """
    radius = 20.5
    h2_radius = 1.1 * radius
    bld_radius = .85 * radius
    hub_pos = hf.Cartesian(0, -2.715, -35.5)

    blade_1 = hf.Cylindrical(bld_radius, 0, 0, hub_pos).to_cartesian()
    blade_2 = hf.Cylindrical(bld_radius, 2 * np.pi / 3, 0, hub_pos).to_cartesian()
    blade_3 = hf.Cylindrical(bld_radius, 4 * np.pi / 3, 0, hub_pos).to_cartesian()

    tip_1 = hf.Cylindrical(radius, 0, 0, hub_pos).to_cartesian()
    tip_2 = hf.Cylindrical(radius, 2 * np.pi / 3, 0, hub_pos).to_cartesian()
    tip_3 = hf.Cylindrical(radius, 4 * np.pi / 3, 0, hub_pos).to_cartesian()

    points, _, _ = hf.uniform_spherical_grid(255)

    def turbine():
        ax.scatter((-blade_1[0], -blade_2[0], -blade_3[0],),
                   (blade_1[1], blade_2[1], blade_3[1],),
                   (-blade_1[2], -blade_2[2], -blade_3[2],),
                   color='k', marker='o', s=50, alpha=1)

        phi = np.linspace(0, 2 * np.pi, 25)

        tower_radius = 1.
        tower_length = np.linspace(0, 35.5, 2)
        phi, tower_length = np.meshgrid(phi, tower_length)

        ax.plot_surface(tower_radius * np.cos(phi), tower_radius * np.sin(phi), tower_length, color='k')

        hub_radius = 1.
        hub_length = np.linspace(-2.715, 2, 2)
        phi, hub_length = np.meshgrid(phi, hub_length)

        ax.plot_surface(hub_radius * np.cos(phi), hub_length, hub_radius * np.sin(phi) - hub_pos[2], color='k')
        ax.scatter(0, -2.715, 35.5, color='k', marker='o', s=50)

        ax.plot((0, -tip_1[0]), (-2.715, tip_1[1]), (35.5, -tip_1[2]), color='r', linewidth=2)
        ax.plot((0, -tip_2[0]), (-2.715, tip_2[1]), (35.5, -tip_2[2]), color='g', linewidth=2)
        ax.plot((0, -tip_3[0]), (-2.715, tip_3[1]), (35.5, -tip_3[2]), color='b', linewidth=2)

    def plot(idx):
        if idx in (0, 3, 4):
            hawc2_grid = np.array([(h2_radius * point + hub_pos).vec for point in points])
            ax.scatter(-hawc2_grid[:, 0], hawc2_grid[:, 1], -hawc2_grid[:, 2], s=4, color='m', alpha=1)

        if idx in (1, ):
            blade_1_grid = np.array([(4 * point + blade_1).vec for point in points])
            ax.scatter(-blade_1_grid[:, 0], blade_1_grid[:, 1], -blade_1_grid[:, 2], color='r', s=.2)

            blade_2_grid = np.array([(4 * point + blade_2).vec for point in points])
            ax.scatter(-blade_2_grid[:, 0], blade_2_grid[:, 1], -blade_2_grid[:, 2], color='g', s=.2)

            blade_3_grid = np.array([(4 * point + blade_3).vec for point in points])
            ax.scatter(-blade_3_grid[:, 0], blade_3_grid[:, 1], -blade_3_grid[:, 2], color='b', s=.2)

        if idx in (2, 3, 4, ):
            blade_1_grid = np.array([(4 * point + blade_1).vec for point in points])
            ax.scatter(-blade_1_grid[:, 0], blade_1_grid[:, 1], -blade_1_grid[:, 2], color='r', s=1)
            for pt in blade_1_grid:
                ax.plot((-blade_1[0], -pt[0]), (blade_1[1], pt[1]), (-blade_1[2], -pt[2]), color='k', linewidth=.5)

            ax.set_xlim(-bld_radius - 10, -bld_radius + 10)
            ax.set_ylim(-2.715 - 10, -2.715 + 10)
            ax.set_zlim(35.5 - 10, 35.5 + 10)

        if idx in (3, 4, ):
            blade_1_points = [point + blade_1 for point in points]
            blade_1_points_plt = [4 * point + blade_1 for point in points]
            for pi, point in enumerate(blade_1_points):
                nabla = np.sum(((point - blade_1) * (blade_1 - hub_pos)).vec) ** 2 - (
                        hub_pos.dist(blade_1) ** 2 - h2_radius ** 2)
                # distance from self to edge of sphere in direction of sphere
                dist = -np.sum(((point - blade_1) * (blade_1 - hub_pos)).vec) + np.sqrt(nabla)
                # point on sphere at end of initial ray
                pt = blade_1 + (point - blade_1) * dist

                ax.plot((-blade_1_points_plt[pi][0], -pt[0]),
                        (blade_1_points_plt[pi][1], pt[1]),
                        (-blade_1_points_plt[pi][2], -pt[2]),
                        color='r', linewidth=.5)

                if idx == 3:
                    ax.scatter(-pt[0], pt[1], -pt[2], color='r', marker='x', s=10)

            ax.set_xlim(-30, 30)
            ax.set_ylim(-30, 30)
            ax.set_zlim(0, 60)

        if idx in (4, ):
            blade_2_grid = np.array([(4 * point + blade_2).vec for point in points])
            for pt in blade_2_grid:
                ax.plot((-blade_2[0], -pt[0]), (blade_2[1], pt[1]), (-blade_2[2], -pt[2]), color='k', linewidth=.5)

            blade_3_grid = np.array([(4 * point + blade_3).vec for point in points])
            for pt in blade_3_grid:
                ax.plot((-blade_3[0], -pt[0]), (blade_3[1], pt[1]), (-blade_3[2], -pt[2]), color='k', linewidth=.5)

            blade_2_points = [point + blade_2 for point in points]
            blade_2_points_plt = [4 * point + blade_2 for point in points]
            for pi, point in enumerate(blade_2_points):
                nabla = np.sum(((point - blade_2) * (blade_2 - hub_pos)).vec) ** 2 - (
                        hub_pos.dist(blade_2) ** 2 - h2_radius ** 2)
                # distance from self to edge of sphere in direction of sphere
                dist = -np.sum(((point - blade_2) * (blade_2 - hub_pos)).vec) + np.sqrt(nabla)
                # point on sphere at end of initial ray
                pt = blade_2 + (point - blade_2) * dist

                ax.plot((-blade_2_points_plt[pi][0], -pt[0]),
                        (blade_2_points_plt[pi][1], pt[1]),
                        (-blade_2_points_plt[pi][2], -pt[2]),
                        color='g', linewidth=.5)

            blade_3_points = [point + blade_3 for point in points]
            blade_3_points_plt = [4 * point + blade_3 for point in points]
            for pi, point in enumerate(blade_3_points):
                nabla = np.sum(((point - blade_3) * (blade_3 - hub_pos)).vec) ** 2 - (
                        hub_pos.dist(blade_3) ** 2 - h2_radius ** 2)
                # distance from self to edge of sphere in direction of sphere
                dist = -np.sum(((point - blade_3) * (blade_3 - hub_pos)).vec) + np.sqrt(nabla)
                # point on sphere at end of initial ray
                pt = blade_3 + (point - blade_3) * dist

                ax.plot((-blade_3_points_plt[pi][0], -pt[0]),
                        (blade_3_points_plt[pi][1], pt[1]),
                        (-blade_3_points_plt[pi][2], -pt[2]),
                        color='b', linewidth=.5)

    for i in range(5):
        # Create the main plot
        fig = plt.figure(i + 1)
        ax = fig.add_subplot(projection='3d')
        # Pre-set the axis limits and aspect ratio
        ax.set_aspect('equal')
        ax.set_xlim(-30, 30)
        ax.set_ylim(-30, 30)
        ax.set_zlim(0, 60)
        ax.set_xlabel('$-x$ (m)')
        ax.set_ylabel('$y$ (m)')
        ax.set_zlabel('$-z$ (m)')

        turbine()
        plot(i)

    plt.show()


if __name__ == '__main__':
    window_plot()

    ground_effect_plot_arntzen()
    ground_effect_plot_embleton()

    psat_plot()
    atm_absorption_plot()

    tool_overview_plots()
    case_plot('validation', '1510_1500_085ms.aur')
    case_plot('validation', '2310_1518_075ms.aur')

    hrtf = hf.MITHrtf()
    hrtf.plot_horizontal()
