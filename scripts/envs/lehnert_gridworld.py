# https://www.codehaven.co.uk/misc/python/using-arrow-keys-with-inputs-python/
import curses
import matplotlib.pyplot as plt
from envs.lehnert_gridworld import LehnertGridworld


env = LehnertGridworld()

screen = curses.initscr()
curses.noecho()
curses.cbreak()
screen.keypad(True)

try:
    while True:
        action = None

        char = screen.getch()
        if char == ord('q'):
            break
        elif char == curses.KEY_RIGHT:
            action = 3
        elif char == curses.KEY_LEFT:
            action = 2
        elif char == curses.KEY_UP:
            action = 0
        elif char == curses.KEY_DOWN:
            action = 1

        if action is not None:
            state, reward, done, _ = env.step(action)
            print("reward: {:.0f}".format(reward))
            plt.imshow(state)
            plt.show()
finally:
    # shut down cleanly
    curses.nocbreak()
    screen.keypad(0)
    curses.echo()
    curses.endwin()
