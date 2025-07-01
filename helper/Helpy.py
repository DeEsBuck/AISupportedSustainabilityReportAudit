from threading import Thread, Event
import datetime as dt
import numpy as np
import timeit
import random
import time
import os


# HelPi.py

def safe_strip(value):
    return str(value).strip() if value is not None else ""


def oba(Object, *optional):
    """
    :param Object:
    :param optional: 'optional star for each following param object
    :return:
    """
    # todo: Iterator Counter for Position and better readability overview, maybe traceback Exception can be a good hint
    if optional:
        print(f'\n>>> #[0]|[{id(Object)}]| {type(Object)} \n=> {Object}')
        for i, s in enumerate(optional):
            print(f'>>> #[{i + 1}]|[{id(s)}]| {type(s)} \n=> {s} ')
    else:
        return print(f'\n>>> #|[{id(Object)}]| {type(Object)} \n=> {Object}')


def randIntList(listsize: int = 1, xposition: int = 10):
    """
    helper Method: Erzeuge eine Liste mit zufälligen Integer Zahlen
    0.0 <= X < 1.0
    :param xposition: Stelle
    :param listsize: größe der Liste
    :return:
    """
    makeList = []
    while listsize > 0:
        listsize = listsize - 1
        if xposition >= 10:
            makeList.append(int(random.random() * xposition))
    
    return makeList


def randInt(xposition: int = 10):
    """
    helper Method: Erzeuge eine beliebig große Zahl
    0.0 <= X < 1.0
    :param xposition: Stelle
    :return:
    """
    return int(random.random() * xposition)


def anSimpleObj(Object, name='anyObject', *optional):
    """
    helper Method: Python Object to analyze and check Variable and Object Attributes
    todo: make it better and for more than one param
    short function names for efficient use. for information use docstrings ;)
    :param name:
    :param Object:
    :return:
    """
    if optional:
        return print(f'[: {name} {id(Object)} | {type(Object)} | {Object} | {optional} :]')
    
    return print(f'[: {id(Object)} | {type(Object)} | {Object} :]')


def anObj(Object, name='anyObject', *optional):
    """
    Substitute print and extend more Meta-Infos for
    Python Object to analyze and check Variable and Object Attributes
    todo: make it better and for more than one param
    short function names for efficient use. for information use docstrings ;)
    :param name:
    :param Object:
    :return:
    """
    # todo: Iterator Counter for Position and better readability overview, maybe traceback Exception can be a good hint
    objName = isinstance(Object, str)
    tpe = colors.fg.orange
    wrtng = colors.fg.green
    undrln = colors.underline
    bld = colors.bold
    objct = colors.fg.blue
    rest = colors.reset
    if optional:
        return print(tpe, f'>>> #|[{len(Object)}]| {id(Object)} | {type(Object)}', objct, f' \n =>\t {Object}\n', wrtng,
                     undrln, bld, f'"{name}":', rest, wrtng, f'(tuple) => {optional}', rest)
    if name != 'anyObject':
        return print(tpe, f'>>> #|[{len(Object)}]| {id(Object)} | {type(Object)}', objct, f' \n =>\t {Object}\n', wrtng,
                     undrln, bld, f'"{name}"', rest)
    return print(tpe, f'>>> #|[{len(Object)}]| {id(Object)} | {type(Object)}', objct, f' \n =>\t {Object}', rest)


def randIntList(listsize: int = 1, xposition: int = 10):
    """
    helper Method: Erzeuge eine Liste mit zufälligen Integer Zahlen
    0.0 <= X < 1.0
    :param xposition: Stelle
    :param listsize: größe der Liste
    :return:
    """
    makeList = []
    while listsize > 0:
        listsize = listsize - 1
        if xposition >= 10:
            makeList.append(int(random.random() * xposition))
    
    return makeList


def randInt(xposition: int = 10):
    """
    helper Method: Erzeuge eine beliebig große Zahl
    0.0 <= X < 1.0
    :param xposition: Stelle
    :return:
    """
    return int(random.random() * xposition)


def clamp(n, minN, maxN):
    """
    Test helper Function to limit numbers to a specific range.
    Thanks to ref.:
    https://stackoverflow.com/questions/5996881/how-to-limit-a-number-to-be-within-a-specified-range-python
    :param maxN:
    :param minN:
    :param n:
    :return:
    """
    if n < minN:
        return minN
    elif n > maxN:
        return maxN
    else:
        return n


def timestamp():
    """
    easy simple Timestamper
    :return:
    """
    dt.datetime.now()
    return (dt.datetime.now() - dt.datetime(1970, 1, 1)).total_seconds()


def execTimeit():
    """
    execute as Callable
    better alternative for timer use to measure code duration is timeit.Timer()
    :return:
    """
    return timeit.timeit(lambda: "-".join(map(str, range(100))), number=10000)


def timer():
    """
    Zeitangaben Dauer
    todo: empfindlichkeit Zeiteinheiten ms oder so und floatingsponts
    :return:
    """
    _start_time = dt.datetime.now()
    _stop_time = dt.datetime.now()
    duration = abs(int((_start_time - _stop_time).total_seconds()))
    print(f'elapsed time: {duration}')
    return duration


class TimerError(Exception):
    """
    A custom exception used to report errors in use of Timer class
    """


class Timer:
    """
    Timer implement timer class with methods for start and endtime to calculate time estimation.
    Runs in parallel and stops with event callback time:
    ============ How to =========
    from timer import Timer
    t = Timer()
    t.start()
    t.stop()  # A few seconds later
    Elapsed time: 3.8191 seconds
    """
    
    def __init__(self):
        self.type = Timer
        self._start_time = None
    
    def start(self):
        """Start a new timer"""
        if self._start_time is not None:
            raise TimerError(f"Timer is running. Use .stop() to stop it")
        
        self._start_time = time.perf_counter()
    
    def stop(self):
        """Stop the timer, and report the elapsed time"""
        if self._start_time is None:
            raise TimerError(f"Timer is not running. Use .start() to start it")
        
        elapsed_time = time.perf_counter() - self._start_time
        self._start_time = None
        print(f"Elapsed time: {elapsed_time:0.4f} seconds")
        return elapsed_time


def randIntFrame(randState=None, randConf=(0, None, None), rowIndex=None, colIndex=None, axis=0):
    """
    - shape(a);
    Return the shape of an array. OR
    - reshape(a, newshape[, order])
    Gives a new shape to an array without changing its data.
    - randint(low, high=None, size=None, dtype=int)
    Return random integers from the "discrete uniform" distribution of the specified dtype in the "half-open" interval [low, high).
    - arange([start,] stop[, step,], dtype=None, *, like=None)
    The built-in range generates :std:doc:`Python built-in integers that have arbitrary size  long>`,
    while numpy.arange produces numpy.int32 or numpy.int64 numbers.
    This may result in incorrect results for large integer values.
    numpy.linspace : Evenly spaced numbers with careful handling of endpoints.
    numpy.ogrid: Arrays of evenly spaced numbers in N-dimensions.
    numpy.mgrid: Grid-shaped arrays of evenly spaced numbers in N-dimensions.
    how-to-partition todo: definition for more accuracy
    :param randConf:
    :param randState:
    :param rowIndex:
    :param colIndex:
    :param axis:
    :return:
    """
    np.random.seed(randState)  # eq: np.random.random_state = 42
    # test case randint() config positions; 0:10:10 => 5 x 2 todo: write algorithm for shape always fits; with, try, exception
    lin = randConf[2]
    dim = randConf[2]
    a = np.random.randint(randConf[0], randConf[1], randConf[2]).reshape(lin, dim)
    if axis == 0:
        rowIndex = np.arange(a.shape[0])
    elif axis == 1:
        colIndex = np.arange(a.shape[1])
    else:
        a.reshape()
    
    return a, rowIndex, colIndex


class colors:
    """Colors class:reset all colors with colors.reset; two
    sub classes fg for foreground
    and bg for background; use as colors.subclass.colorname.
    i.e. colors.fg.red or colors.bg.greenalso, the generic bold, disable,
    underline, reverse, strike through,
    and invisible work with the main class i.e. colors.bold
    Usage e.g.:
        print(colors.bg.green, "SKk", colors.fg.red, "Amartya")
        print(colors.bg.lightgrey, "SKk", colors.fg.red, "Amartya")
    """
    reset = '\033[0m'
    bold = '\033[01m'
    disable = '\033[02m'
    underline = '\033[04m'
    reverse = '\033[07m'
    strikethrough = '\033[09m'
    invisible = '\033[08m'
    
    class fg:
        black = '\033[30m'
        red = '\033[31m'
        green = '\033[32m'
        orange = '\033[33m'
        blue = '\033[34m'
        purple = '\033[35m'
        cyan = '\033[36m'
        lightgrey = '\033[37m'
        darkgrey = '\033[90m'
        lightred = '\033[91m'
        lightgreen = '\033[92m'
        yellow = '\033[93m'
        lightblue = '\033[94m'
        pink = '\033[95m'
        lightcyan = '\033[96m'
    
    class bg:
        black = '\033[40m'
        red = '\033[41m'
        green = '\033[42m'
        orange = '\033[43m'
        blue = '\033[44m'
        purple = '\033[45m'
        cyan = '\033[46m'
        lightgrey = '\033[47m'


def custom_fill(text, width, fill_char):
    """
    Beispielaufruf der Funktion:
    text = "123"
    width = 6
    fill_char = "*"
    result = hepy.custom_fill(text, width, fill_char)
    print(result)
    :param text:
    :param width:
    :param fill_char:
    :return:
    """
    # Berechne die Anzahl der Zeichen, die eingefügt werden müssen
    fill_count = max(0, width - len(text))
    
    # Füge die benutzerdefinierten Füllzeichen vor den Text ein
    filled_text = fill_char * fill_count + text
    
    return filled_text


def replace_list(input_list, replacement_value, times):
    # Replace the value at the given index 'times' times
    for _ in range(times):
        input_list[_] = replacement_value
        _ += 1


def fill_list(input_list, index, fill_value):
    # Fill the rest of the mylist with the fill_value
    while index < len(input_list):
        input_list[index] = fill_value
        index += 1


# Code to execute in an independent thread
def countdown(n, started_evt):
    """
    # Import modules
    from threading import Thread, Event

    # Create the event object that
    # will be used to signal startup
    started_evt = Event()

    # Launch the thread and pass the startup event
    print('Launching countdown')
    t = Thread(target = countdown, args =(10, started_evt))
    t.start()

    # Wait for the thread to start
    started_evt.wait()
    print('countdown is running')
    :param n:
    :param started_evt:
    :return:
    """
    # Code to execute in an independent thread
    print('countdown starting')
    started_evt.set()
    
    while n > 0:
        print('T-minus', n)
        n -= 1
        time.sleep(5)


class MyException(Exception): pass


class KeyloggerHelper:
    def __init__(self, keys=None):
        if keys is None:
            keys = []
        self.keys = keys
    
    def key_event_handler(self, list_mods=0):
        """
        keylogger, event starts thread, listen until user 'esc' key triggered
        :param list_mods:
        :return:
        """
        
        def on_press(key):
            try:
                if key == Key.tab:
                    print("good")
                if key != Key.tab:
                    print("try again")
                
                print('alphanumeric key {0} pressed'.format(key.char))
            except AttributeError:
                print('special key {0} pressed'.format(key))
        
        def on_release(key):
            print('{0} released'.format(key))
            if key == keyboard.Key.esc:
                # Stop listener
                return False
        
        if list_mods == 0:
            def on_release(key):
                print('{0} released'.format(key))
                if key == keyboard.Key.esc:
                    raise MyException(key)
            
            # To read a single event, use the following code:
            # The event listener will be running in this block
            with keyboard.Events() as events:
                # Block at most one second
                event = events.get(1.0)
                if event is None:
                    print('You did not press a key within one second')
                else:
                    print('Received event {}'.format(event))
        elif list_mods == 1:
            # To iterate over keyboard events, use the following code:
            # The event listener will be running in this block
            with keyboard.Events() as events:
                for event in events:
                    if event.key == keyboard.Key.esc:
                        break
                    else:
                        print('Received event {}'.format(event))
        elif list_mods == 2:
            # Collect all event until released
            with Listener(on_press=on_press, on_release=on_release) as listener:
                listener.join()
        elif list_mods == 3:
            # Collect events until released
            with keyboard.Listener(
                    on_press=on_press) as listener:
                try:
                    listener.join()
                except MyException as e:
                    print('{0} was pressed'.format(e.args[0]))
        else:
            # ...or, in a non-blocking fashion:
            listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            listener.start()
    
    def kellogs_list(self):
        """
        Keys[char] writes String file from keylogger Stream, event starts thread, listen until user 'esc' key triggered
        :return:
        """
        
        def on_press(key):
            self.keys.append(key)
            self.write_file(self.keys)
            try:
                print('alphanumeric key {0} pressed'.format(self.char))
            except AttributeError:
                print('special key {0} pressed'.format(key))
        
        def write_file(key):
            with open('log.txt', 'w') as f:
                for key in self.keys:
                    # removing ''
                    k = str(key).replace("'", "")
                    f.write(k)
                    # explicitly adding a space after
                    # every keystroke for readability
                    f.write(' ')
        
        def on_release(key):
            print('{0} released'.format(key))
            if key == Key.esc:
                # Stop listener
                return False
        
        with Listener(on_press=on_press, on_release=on_release) as listener:
            listener.join()


# todo: WRITE a function for this? if ENUM not already offers some kind of a collector...
def gaussum(n):
    """
    Gauss sum=(((0+1)+2)+3)+…+n
    :return:
    """
    if n == 0:
        return
    else:
        return n + gaussum(n - 1)
