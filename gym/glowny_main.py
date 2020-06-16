#!/usr/bin/env python
import sys, gym
import cv2
import numpy as np
import seaborn as sns; sns.set()
import math
from sklearn.decomposition import PCA
from pydstarlite import grid
from pydstarlite.dstarlite import DStarLite


env = gym.make('Asteroids-v0' if len(sys.argv) < 2 else sys.argv[1])

if not hasattr(env.action_space, 'n'):
    raise Exception('Keyboard agent only supports discrete action spaces')
ACTIONS = env.action_space.n
SKIP_CONTROL = 0    # Use previous control decision SKIP_CONTROL times, that's how you
                    # can test what skip is still usable.

human_agent_action = 0
set_end_point = 0
human_wants_restart = False
human_sets_pause = False


def key_press(key, mod):
    global human_agent_action, human_wants_restart, human_sets_pause, set_end_point
    if key==0xff0d: human_wants_restart = True
    if key==32: human_sets_pause = not human_sets_pause
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    # human_agent_action = a
    set_end_point = a


def key_release(key, mod):
    global human_agent_action, set_end_point
    a = int( key - ord('0') )
    if a <= 0 or a >= ACTIONS: return
    if human_agent_action == a:
        human_agent_action = 0


env.render()
env.unwrapped.viewer.window.on_key_press = key_press
env.unwrapped.viewer.window.on_key_release = key_release


def rollout(env):
    global human_agent_action, human_wants_restart, human_sets_pause, set_end_point
    human_wants_restart = False
    obser = env.reset()
    skip = 0
    total_reward = 0
    total_timesteps = 0
    lewo = False
    prawo = False
    zapamietywajka = [0, 0]
    a = 0
    delay = 0

    # PCA _init_
    pca = PCA(n_components=2)

    while 1:
        if not skip:
            # a = human_agent_action
            total_timesteps += 1
            skip = SKIP_CONTROL
        else:
            skip -= 1
        # print(a)
        obser, r, done, info = env.step(a)          # co step pobieramy obserwer - generowany obraz gry
        obrazek = np.zeros((210, 160), np.uint8)    # tutaj tworzymy własny obraz ala mapa zajetosci (210x160)
        a = 0

        # statek_pixele
        statek_pxl = []
        y = x = (0, 0)

        # D*-Lite tworzenie GridMap
        g = grid.SquareGrid(16, 21)                 # tutaj tworzymy GRID dla D*Lite (21x16)

        # tworzenie occupancy map
        for i in range(15, obser.shape[0]):
            for j in range(obser.shape[1]):
                if i == 15:
                    obrazek[i][j] = 255             # tutaj oznaczamy granice gry

                if obser[i][j][0] != obrazek[i][j]:       # zaznaczamy meteortyny na mapie zajetosci
                    j_floor = math.floor(j/10)
                    i_floor = math.floor(i/10)
                    obrazek[i][j] = 255
                    g.walls.add((j_floor, i_floor))       # dodajemy meteoryt do GRID map

                if obser[i][j][1] == 128:               # zaznaczamy statek na mapie zajetosci kolorem 200
                    obrazek[i][j] = 200
                    statek_pxl.append([j, i])

        try:                                            # to było potrzebne ze wzgledu na problem regresji liniowej
            pca.fit(statek_pxl)
            if pca.components_[0][0] == 0. and round(zapamietywajka[0], 2) == 0.47:
                prawo = True

            if pca.components_[0][0] == 0. and round(zapamietywajka[0], 2) == -0.47:
                lewo = True

            if round(pca.components_[0][0], 2) == 0.47 or round(pca.components_[0][0], 2) == -0.47:
                lewo = False
                prawo = False
        except:
            pass

        try:                                                # tutaj liczymy polozenie statku x
            # x to coordynaty (x, y) statku
            x = (int(pca.mean_[0]), int(pca.mean_[1]))

            # y to kierunek zwrotu statku                   # tutaj orientacja statku
            if lewo is True:
                y = (int(pca.mean_[0] + 40 * -1), int(pca.mean_[1] + 40 * 0))
            if prawo is True:
                y = (int(pca.mean_[0] + 40 * 1), int(pca.mean_[1] + 40 * 0))
            if lewo is not True and prawo is not True:
                y = (int(pca.mean_[0] + 40 * pca.components_[0][0]), int(pca.mean_[1] + 20 * pca.components_[0][1]))

            zapamietywajka = pca.components_[0]
        except:
            pass

        # nowy D*-Lite                                                  # ustawienie punktu startowego algorytmu
        # set start point = x,y stateku
        start = (math.floor(x[0] / 10), math.floor(x[1] / 10))
        # set end poin (1, 2, 3 or 4)
        if set_end_point == 0: end = (5, 10)
        if set_end_point == 1: end = (0, 2)
        if set_end_point == 2: end = (15, 2)
        if set_end_point == 3: end = (0, 20)
        if set_end_point == 4: end = (15, 20)

        dstar = DStarLite(g, start, end)                                # D* - Lite
        try:                                                            # tutaj wyznaczana jest sciezka
            path = [p for p, o, w in dstar.move_to_goal()]
            for i in path:
                obrazek[(i[1] * 10):(i[1] * 10) + 5, (i[0] * 10):(i[0] * 10) + 5] = 80

            # nadawanie trajektorii                                     # obliczanie trajektorii statku
            sciezka_x = path[3][0] * 10                                 # na podstawie Path
            sciezka_y = path[3][1] * 10
            koncowka_x = y[0]
            koncowka_y = y[1]

            if delay % 2 == 0:                                          # wysylanie komend do statku z delay
                if(koncowka_x + 10 > sciezka_x and koncowka_x - 10 < sciezka_x and
                        koncowka_y + 10 > sciezka_y and koncowka_y - 10 < sciezka_y):
                    # print('CALA NAPRZOD GENERALE!')
                    a = 2
                else:
                    if x[0] < sciezka_x:
                        # print('KRAKEN PO PRAWEJ')
                        if koncowka_y < sciezka_y:
                            # print('STERBURTA W PRAWO!')
                            if a != 3:
                                a = 3
                        elif koncowka_y > sciezka_y:
                            # print('STERBURTA W LEWO!')
                            if a != 4:
                                a = 4
                    else:
                        # print('KRAKEN PO LEWEJ')
                        if koncowka_y > sciezka_y:
                            # print('STERBURTA W LEWO!')
                            if a != 3:
                                a = 3
                        elif koncowka_y < sciezka_y:
                            # print('STERBURTA W PRAWO!')
                            if a != 4:
                                a = 4
        except:
            pass
        delay += 1

        if(x[0] + 10 > end[0] * 10 and x[0] - 10 < end[0] * 10 and
                x[1] + 10 > end[1] * 10 and x[1] - 10 < end[1] * 10):
            print('IM FINE')
            a = 5

        # wizualizacja                                                          # wizualizacja
        cv2.line(obrazek, x, y, color=255)
        cv2.circle(obrazek, (end[0] * 10, end[1] * 10), 10, color=255)
        obrazek2 = cv2.resize(obrazek.copy(), (0, 0), fx=3, fy=3)
        cv2.imshow('moon', obrazek2)
        cv2.waitKey(1)

        if r != 0:
            print("reward %0.3f" % r)
        total_reward += r
        window_still_open = env.render()
        if window_still_open is False: return False
        if done: break
        if human_wants_restart: break
        while human_sets_pause:
            env.render()                                                        # glowny RENDER
        #     time.sleep(0.1)
        # time.sleep(0.1)
    print("timesteps %i reward %0.2f" % (total_timesteps, total_reward))


# print("ACTIONS={}".format(ACTIONS))
# print("Press keys 1 2 3 ... to take actions 1 2 3 ...")
# print("No keys pressed is taking action 0")
print('\n- Asteroids-v0 with D*-Lite Algorithm -')
print('    ZMIANA END POINT:')
print('    1 - lewy gory rog\n    2 - prawy gory rog')
print('    3 - lewy dolny rog\n    4 - prawy dolny rog')

while 1:
    window_still_open = rollout(env)
    if window_still_open == False: break