from pyswarms.single.global_best import GlobalBestPSO
import matplotlib.pyplot as plt
import numpy as np


# Struktury
pop_b = np.array([
    [0.1, 0.2, 0.1, 0.3],
    [0.2, 0.1, 0.2, 0.2],
    [0.3, 0.3, 0.1, 0.1],
    [0.2, 0.2, 0.3, 0.3]
])
pop_a = np.array([
    [500, 550, 600, 650],
    [520, 580, 620, 680],
    [540, 610, 640, 710],
    [560, 640, 660, 740]
])

pod_d = np.array([
    [0.1, 0.2, 0.1, 0.3], 
    [0.2, 0.1, 0.2, 0.2], 
    [0.3, 0.3, 0.1, 0.1], 
    [0.2, 0.2, 0.3, 0.3]
])
pod_c = np.array([
    [10, 20, 30, 40],
    [15, 25, 35, 45], 
    [20, 30, 40, 50], 
    [25, 35, 45, 55]
])


# koszt dostosowania
przystosowanie = np.array([
    [15000, 12000, 13000, 16000],
    [17000, 14000, 11000, 18000],
    [12000, 16000, 15000, 17000],
    [18000, 13000, 14000, 15000]
])

# jakosc
jakosc = np.array([
    [30, 20, 25, 15],
    [35, 18, 30, 12],
    [22, 38, 20, 28],
    [40, 10, 35, 18]
])

# pracownicy
pracochlonnosc = np.array([
    [2000, 4000, 6000, 8000],
    [4000, 6000, 8000, 10000],
    [6000, 8000, 10000, 12000],
    [8000, 10000, 12000, 14000]
])
koszt_pracownika = np.array([
    [28, 30, 31, 29],
    [25, 28, 31, 35],
    [26, 29, 32, 33],
    [27, 30, 34, 34]
])

# koszt materialow
koszt_materialow = np.array([
    [5, 7, 10, 12],
    [8, 9, 14, 16],
    [11, 12, 18, 20],
    [14, 15, 22, 24]
])

# wsparcie
wsparcie_produktu = np.array([
    [30, 35, 40, 45],
    [35, 40, 45, 50],
    [40, 45, 50, 55],
    [45, 50, 55, 60]
])

# magazyn
magazyn_zapotrzebowanie = 2000
magazyn_cena = 5

# budzet
budzet = 1000000

koszt_sprzedazy_cust = None

# oblicza cenę ze wzoru na krzywą podaży P = (Qs - c) / d
def ceny(X):
    P = np.divide(np.round(X) - pod_c, pod_d)
    return P  

# koszt zakupu materiałów ZR
def koszt1(X):
    mask = X >= 500
    uwzgledniony_rabat = np.where(mask, X * koszt_materialow, X * koszt_materialow * 0.9)
    suma_wierszy = np.sum(uwzgledniony_rabat, axis=(1, 2))
    return suma_wierszy

# koszt pracowników ZR
def koszt2(X):
    lp = np.round(np.multiply(np.round(X), pracochlonnosc) / (250 * 8))
    suma_wierszy = np.sum(np.multiply(lp, koszt_pracownika), axis=(1, 2))
    return suma_wierszy

# koszt przystosowania parku maszynowego ZR
def koszt3(X):
    nowa_macierz = np.where(np.round(X) > 0, przystosowanie, 0)
    suma_wierszy = np.sum(nowa_macierz, axis=(1, 2))
    return suma_wierszy


# koszt wsparcia produktu
def koszt4(X, ceny_sprz):
    wybrane_elementy = np.where(np.round(X) > 0, jakosc, 0)
    wynik_jakosciowy = np.sum(wybrane_elementy, axis=(1, 2))
    pop_a_jakosc = np.tile(pop_a, (X.shape[0], 1, 1))

    # Warunkowo dodaj 200 do elementów, jeśli wartości wyniku jakosciowego > 160
    pop_a_jakosc[wynik_jakosciowy > 160] += 200

    il_pop = pop_a_jakosc - pop_b * ceny_sprz
    roznica = np.maximum(X - il_pop, 0)
    suma_wierszy = np.sum(np.multiply(np.round(roznica)**2, wsparcie_produktu), axis=(1, 2))
    return suma_wierszy, il_pop

# koszt magazyn ZR
def koszt5(X, il_pop):
    roznica = X - il_pop
    suma_wierszy = np.sum(np.where(np.round(roznica) > 0, roznica, 0), axis=(1, 2))
    wektor_wynikowy = np.multiply(np.round(suma_wierszy), magazyn_cena)
    return wektor_wynikowy

# koszt kredytowania ZR
def koszt6(X, k1, k2, k3, k4, k5):
    new_budzet =  np.full(X.shape[0], budzet)
    suma_kar = k1 + k2 + k3 + k4 + k5
    nowy_wektor = np.where(suma_kar > new_budzet, new_budzet * suma_kar * 1.12, 0)
    return nowy_wektor

# zysk ZR
def zyski_sprzedaz(X, cena_sprz):
    if koszt_sprzedazy_cust is None:
        suma_wierszy = np.sum(X * cena_sprz, axis=(1, 2))
    else:
        suma_wierszy = np.sum(X * koszt_sprzedazy_cust, axis=(1, 2))

    return suma_wierszy


# Funkcja celu
def cost_fun(Xf):
    X = Xf.reshape((Xf.shape[0], 4, 4))
    ceny_sprz = ceny(X)
    k1 = koszt1(X)
    k2 = koszt2(X)
    k3 = koszt3(X)
    k4, il_pop = koszt4(X, ceny_sprz)
    k5 = koszt5(X, il_pop)
    k6 = koszt6(X, k1, k2, k3, k4, k5)
    zysk = zyski_sprzedaz(X, ceny_sprz)
    return  -(zysk - (k1 + k2 + k3 + k4 + k5 + k6))

# PySwarms
def run_pso(c1=0.7, c2=0.7, w=0.7, particles=50, iterations=200, init_pos=None):
    options = {'c1': c1, 'c2': c2, 'w': w}
    n_particles = particles
    iters = iterations
    dimensions = 16
    bounds = (np.zeros(16), np.full(16, 10000))
    init_pos = np.ones((n_particles, dimensions))
    
    optimizer = GlobalBestPSO(n_particles=n_particles, dimensions=dimensions, options=options, bounds=bounds, init_pos=init_pos)
    cost, pos = optimizer.optimize(cost_fun, iters=iters)
    best_position_matrix = np.array(pos).reshape((4, 4))
    return cost, optimizer, best_position_matrix

# wektor predkosci
def get_pso_result(c1=0.7, c2=0.7, w=0.7, particles=50, iterations=200, koszt_sprzedazy=None):
    global koszt_sprzedazy_cust
    koszt_sprzedazy_cust = koszt_sprzedazy
    test1_cost = []
    test1_hist = []
    test1_pos = []
    for i in range(10):
        temp_cost_test1, optimizer, pos_mat  = run_pso(w=w, c1=c1, c2=c2, particles=particles, iterations=iterations)
        temp_hist_test1 = optimizer.cost_history
        test1_pos.append(pos_mat)
        test1_cost.append(temp_cost_test1)
        test1_hist.append(temp_hist_test1)


    opposite_sign_list = [round(-x) for x in test1_cost]
    print("---Koszt: ", opposite_sign_list)
    print("---Max: ", max(opposite_sign_list))
    print("---Min: ", min(opposite_sign_list))
    print("---Średnia: ", np.round(np.mean(opposite_sign_list)))
    print("---Mediana: ", np.median(opposite_sign_list))
    print("---Odchylenie st: ", np.round(np.std(opposite_sign_list)))
    print("---Pozycja: ", np.round(test1_pos))
    for i in test1_hist:
        plt.plot([ -x for x in i], linewidth=3)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlabel('Iteracje', fontsize=16)
    plt.ylabel('Wartość funkcji kosztu', fontsize=18)
    plt.ticklabel_format(axis='y', style='plain') 
    plt.show()

