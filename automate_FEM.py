from automan.api import Problem, Automator
from matplotlib import pyplot as plt
import numpy as np
from automan.api import Simulation
from automan.api import mdict, opts2path


class FEM(Problem):
    def get_name(self):
        return 'fem'

    def setup(self):
        opts = mdict(Mat=["210000,0.28", "125000,0.30", "100000,0.34",
                          "71000,0.32"],
                     procedure=['Plane_strain', 'Plane_stress'])
        self.cases = [
            Simulation(root=self.input_path(opts2path(kw)),
                       base_command='python 2D_FEM.py \
                       -W 20 -H 100 -noex 20 -noey 100 -f 500 \
                       --output-dir $output_dir',
                       **kw
                       )
            for kw in opts
        ]

    def run(self):
        self.make_output_dir()
        self.output_path(('data.npz'))
        data_PE = []
        opts1 = mdict(Mat=["210000,0.28", "125000,0.30", "100000,0.34",
                           "71000,0.32"],
                      procedure=['Plane_strain'])
        for kw in opts1:
            stdout = self.input_path(str(opts2path(kw)), 'stdout.txt')
            with open(stdout) as f:
                values = [float(x) for x in f.read().split()]
                data_PE.append(values)

        data_PS = []
        opts2 = mdict(Mat=["210000,0.28", "125000,0.30", "100000,0.34",
                           "71000,0.32"], procedure=['Plane_stress'])
        for kw in opts2:
            stdout = self.input_path(str(opts2path(kw)), 'stdout.txt')
            with open(stdout) as f:
                values = [float(x) for x in f.read().split()]
                data_PS.append(values)

        data_PE = np.asarray(data_PE)
        data_PS = np.asarray(data_PS)

        x = data_PE[:, :2]
        y_PE = data_PE[:, 2]
        y_PS = data_PS[:, 2]

        fname = self.output_path('perf.npz')
        np.savez(fname, x=x, y_PE=y_PE, y_PS=y_PS)

        data = np.load(fname, 'perf.npz')
        A = data['x']
        E = A[:, 0]
        Disp_PE = data['y_PE']
        Disp_PS = data['y_PS']

        plt.plot(E, Disp_PE, color='r', label='PE')
        plt.plot(E, Disp_PS, color='b', label='PS')
        plt.xlabel('E')
        plt.ylabel('Disp')
        plt.title('2D_FEM')
        plt.legend()
        plt.savefig(self.output_path('perf.png'))
        plt.close()


if __name__ == '__main__':
    automator = Automator(
        simulation_dir='outputs',
        output_dir='manuscript/figures',
        all_problems=[FEM]
    )
    automator.run()
