import numpy as np
import random
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from numba import njit
import time
from IPython.display import HTML


def siatka(n):
  if ('-' in n and n.lstrip('-').isdecimal()):
    raise ValueError("Rozmiar siatki musi być większy od 0.")
  if n.isdecimal() == False:
    raise TypeError("Rozmiar siatki musi być liczbą całkowitą.")
  n = int(n)
  return (np.random.randint(0, 2, (n,n))*2-1).astype(float)

@njit
def suma_sasiad(siatka, x, y, n): #x jest w poziomie, y w pionie
  suma = 0
  for i in range(3):
    for j in range(3):
      #print(a[i,j])
      suma += siatka[(y+i-1) % n, (x+j-1) % n]
  suma -= siatka[y,x]
  return suma

@njit
def suma_calk(siatka):
  n = len(siatka)
  suma = 0
  for i in range(n):
    for j in range(n):
      suma += suma_sasiad(siatka, i, j, n)
  return suma

@njit
def E(siatka, J, B):
  if not isinstance(B, float):
    raise TypeError("B musi być liczbą zmiennoprzecinkową.")
  if not isinstance(J, float):
    raise TypeError("J musi być liczbą zmiennoprzecinkową.")
  n = len(siatka)
  B = float(B)
  J = float(J)
  return - B * siatka.sum() - J * suma_calk(siatka)

@njit
def dE(siatka, cords, J, B):
  n = len(siatka)
  x = cords[1]
  y = cords[0]
  return 2 * siatka[y, x] * (J * suma_sasiad(siatka, x, y, n) + B)

@njit
def makro(siatka, J, B, beta, E):
  if not isinstance(beta, float):
    raise TypeError("Beta musi być liczbą zmiennoprzecinkową.")
  if beta < 0:
    raise ValueError("Beta musi być większe lub równe 0.")
  if not isinstance(B, float):
    raise TypeError("B musi być liczbą zmiennoprzecinkową.")
  if not isinstance(J, float):
    raise TypeError("J musi być liczbą zmiennoprzecinkową.")
  n = len(siatka)
  H = E
  siat = np.copy(siatka)
  for i in range(n**2):
      cords = [random.randint(0,n-1),random.randint(0,n-1)]
      d = dE(siat, cords, J, B)
      if d < 0:
          siat[cords[0]][cords[1]] *= -1
          H += d
      else:
          if np.exp(-beta * d) >= random.random():
              siat[cords[0]][cords[1]] *= -1
              H += d
  M = siat.sum() / n**2
  return siat, H, M

@njit
def Sim(siat, J, B, beta, N, E):
  if (N <= 0):
    raise ValueError("Rozmiar siatki musi być większy od 0.")
  if not isinstance(N, int):
    raise TypeError("Rozmiar siatki musi być liczbą całkowitą.")
  if not isinstance(beta, float):
    raise TypeError("Beta musi być liczbą zmiennoprzecinkową.")
  if beta < 0:
    raise ValueError("Beta musi być większe lub równe 0.")
  if not isinstance(B, float):
    raise TypeError("B musi być liczbą zmiennoprzecinkową.")
  if not isinstance(J, float):
    raise TypeError("J musi być liczbą zmiennoprzecinkową.")

  n = len(siat)
  Siat = siat
  Mak = [Siat]
  Energia = [E]
  Magnetyzacja = [Siat.sum() / n**2]
  for i in range(N):
    Siat, Ene, Mag = makro(Siat, J, B, beta, Energia[i])
    Mak.append(Siat)
    Energia.append(Ene)
    Magnetyzacja.append(Mag)
  return Mak, Energia, Magnetyzacja

def animacja(Stany, show, nazwa):
  n = len(Stany[0])
  N = len(Stany)
  rows, cols = n, n

  fig = plt.figure(figsize=(4, 4))
  fig.subplots_adjust(0, 0, 1, 1)
  plt.axis('off')
  im = plt.imshow(Stany[0], cmap='gray', interpolation='nearest', vmin=-1, vmax=1)

  def animate(frame):
      im.set_array(Stany[frame])
      return [im]

  ani = animation.FuncAnimation(fig, animate, frames=N, interval=100, blit=True, repeat=True)

  if nazwa is not None:
    ani.save(f"{nazwa}.gif", writer="pillow", fps=10)

  if show:
    plt.show()
  plt.close()

  #return HTML(ani.to_jshtml())