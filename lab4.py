import ising as i
import matplotlib.pyplot as plt
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--size', default='100')
parser.add_argument('--J', default=1.0, type=float)
parser.add_argument('--B', default=0.0, type=float)
parser.add_argument('--beta', default=0.4, type=float)
parser.add_argument('--N', default=500, type=int)
parser.add_argument('--show_animation', action='store_true', default=False)
parser.add_argument('--save_animation', type=str, default=None, help='Zapisuje animacje do pliku .gif')
parser.add_argument('--save_magnetization', type=str, default=None, help='Zapisuje magnetyzację do pliku')

parser.error = lambda e: parser.exit(0, f'Błąd przy parsowaniu: {e}\n')

args = parser.parse_args()

try:
    siat = i.siatka(args.size)
except (ValueError,TypeError) as e:
    parser.exit(f'Błąd przy tworzeniu siatki: {e}')

# J = args.J
# B = args.B
# beta = args.beta
# N = args.N

try:
    H = i.E(siat, args.J, args.B)
except (ValueError, TypeError) as e:
    parser.exit(f'Błąd przy obliczaniu energii: {e}')
try:
    Stany, Energia, Magnetyzacja = i.Sim(siat, args.J, args.B, args.beta, args.N, H)
except (ValueError, TypeError) as e:
    parser.exit(f'Błąd przy inicjalizacji symulacji: {e}')

plt.subplot(2,1,1)
plt.plot(range(args.N+1), Energia)
plt.xlabel('Makrokrok')
plt.ylabel('Energia')
plt.subplot(2,1,2)
plt.plot(range(args.N+1), Magnetyzacja)
plt.ylim(-1.1, 1.1)
plt.xlabel('Makrokrok')
plt.ylabel('Magnetyzacja')
plt.tight_layout()
plt.show()

if args.save_magnetization is not None:
    if not args.save_magnetization.endswith('.txt') and not args.save_magnetization.endswith('.csv'):
        raise OSError('OSError: Nazwa pliku musi kończyć się na .txt lub .csv')
        
    np.savetxt(args.save_magnetization, Magnetyzacja)

i.animacja(Stany, args.show_animation, args.save_animation)