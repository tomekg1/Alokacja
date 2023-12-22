import PSO
import numpy as np

def test1():
    PSO.get_pso_result(w=0.7, c1=0.7, c2=0.7)

def test2():
    PSO.get_pso_result(w=0.9, c1=1.0, c2=1.0)

def test3():
    PSO.get_pso_result(w=0.8, c1=0.5, c2=0.9)

def test4():
    PSO.get_pso_result(particles=20, iterations=100)

def test5():
    PSO.get_pso_result(particles=50, iterations=150)

def test6():
    PSO.get_pso_result(particles=30, iterations=200)

def test7():
    custom_sprz = np.array([
    [10, 20, 10000, 5],
    [10, 20, 5, 10],
    [10, 20, 5, 5],
    [10, 20, 5, 5]
])
    PSO.get_pso_result(koszt_sprzedazy=custom_sprz)

def test8():
    custom_sprz = np.array([
    [10, 20, 5, 5],
    [10, 10000, 5, 5],
    [10, 20, 5, 5],
    [10, 20, 5, 10000]
])
    PSO.get_pso_result(koszt_sprzedazy=custom_sprz)

test1()
#test2()
#test3()
#test4()
#test5()
#test6()
#test7()
#test8()