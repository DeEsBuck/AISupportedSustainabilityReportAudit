import os
import logging as log
import helper.Helpy
from lmstudio import (
    BaseModel,
    ModelSchema
)
from typing import Protocol
import msgspec
from enum import IntFlag, Enum, Flag, auto, EnumType
from colorama import init, Fore, Back, Style


class Colors:
    """
    With usage of the Module Colorama to provide cros platform color Support
    # Example usage:
    print(Colors.BG.GREEN.value, Colors.FG.RED.value, "Amartya", Colors.RESET.value)
    if init(autoreset=True) then RESET = '' # empty String else Style.RESET_ALL
    """
    
    init(autoreset=False, convert=None, strip=None, wrap=True)
    
    RESET = Style.RESET_ALL
    BOLD = Style.BRIGHT
    UNDERLINE = Style.BRIGHT + Back.BLACK
    
    class FG:
        BLACK = Fore.BLACK
        WHITE = Fore.LIGHTWHITE_EX
        DARK_GREY = Fore.LIGHTBLACK_EX
        LIGHT_GREY = Fore.WHITE
        RED = Fore.RED
        LIGHT_RED = Fore.LIGHTRED_EX
        GREEN = Fore.GREEN
        LIGHT_GREEN = Fore.LIGHTGREEN_EX
        ORANGE = Fore.YELLOW
        YELLOW = Fore.LIGHTYELLOW_EX
        BLUE = Fore.BLUE
        LIGHT_BLUE = Fore.LIGHTBLUE_EX
        CYAN = Fore.CYAN
        LIGHT_CYAN = Fore.LIGHTCYAN_EX
        PURPLE = Fore.MAGENTA
        PINK = Fore.LIGHTMAGENTA_EX
    
    class BG:
        BLACK = Back.BLACK
        WHITE = Back.WHITE
        DARK_GREY = Back.LIGHTBLACK_EX
        RED = Back.RED
        GREEN = Back.GREEN
        ORANGE = Back.YELLOW
        BLUE = Back.BLUE
        CYAN = Back.CYAN
        PURPLE = Back.MAGENTA
    
    @staticmethod
    def split_rainbow_list(color: list = None):
        """
        Switches in odd and even through the colors and print text
        Optional: [_l_color_list, _l_color_list]
        :param color:
        :return:
        """
        if color is None:
            _color_list = [color for color in Colors.FG.__dict__.values() if isinstance(color, str)]
            _color_list.pop(0)  # remove '__main__
        else:
            _color_list = [colors for colors in color[0:] if
                           isinstance(color, (list, str)) and len(color) >= 2]
        _l_color_list = []
        _r_color_list = []
        _obj = {}
        _mod = lambda x: x % 2 == 1
        _m = {0: 'EVEN', 1: 'ODD'}
        for pos, value in enumerate(_color_list, 0):
            _ark = _m.get(_mod(pos))
            _obj[pos] = [int(_mod(pos)), _ark, value]
            # split _list items in two parts _l_r_ usage
            if _mod(pos):
                _l_color_list.append(value)
            else:
                _r_color_list.append(value)
        
        # Extra: Fair sorted into dict
        mod_color_list = _obj
        # Quick and Dirty split
        color_list = [_l_color_list, _r_color_list]
        # unpack 1st lvl, left of Q&D split
        _traversed = Colors.unpack_list_items(color_list)
        # unpack left
        _l_traversed = Colors.unpack_list_items(color_list[0])
        # then right
        _r_traversed = Colors.unpack_list_items(color_list[1])
        
        if FlagSwitch.VERBOSE == 4:
            log.info(f'#00|:{_color_list}')
        if FlagSwitch.VERBOSE == 5:
            loog.info(f'||FULL[[0][1]]: {_traversed} | {Colors.RESET}split_rainbow_list')
            log.info(f'||LEFT_[0]||||: {_l_traversed} | {Colors.RESET}split_rainbow_list')
            log.info(f'||RIGHT[1]||||: {_r_traversed} | {Colors.RESET}split_rainbow_list')
        if FlagSwitch.VERBOSE == 5:
            log.info(f'#01|:{color_list}:| split_rainbow_list')
            log.info(f'|MOD-VIEW|:{mod_color_list}:| split_rainbow_list')
        
        # return Q&D l-r-list
        return color_list
    
    @staticmethod
    def unpack_list_items(mylist: list = None):
        """
        Unpack list trough, simple recursive list traversing
        :param mylist:
        :return:
        """
        if not mylist or None:
            return
        _mylist = mylist[0]
        if FlagSwitch.VERBOSE == 5:
            if isinstance(_mylist, (list, str)) and len(mylist) >= 1:
                inmylist = f'{mylist}'
                if len(_mylist) > 1:
                    inmylist = f'{_mylist}▸▏{Colors.RESET}{inmylist}{_mylist}◾{Colors.RESET}'
                log.info(inmylist)
        Colors.unpack_list_items(mylist[1:])
        return mylist
    
    @staticmethod
    def color_rainbow_dict(mylist, *color):
        """
        Recursive Dictionary List traversal, coloring line per line in specified or default rainbow Color.
        E.g.:
        for rain, bow in enumerate(rainbow):
            print(bow[0], rain, bow[1:])
        :param mylist:
        :param color:
        :return:
        """
        if not mylist:
            return
        else:
            if color:
                _color = color[0:1][0]
            else:
                _color = [color for color in Colors.FG.__dict__.values() if isinstance(color, str)]
                _color.pop(0)  # remove '__main__
        _filter, sorted, _newlist, newlisted = [], [], [], []
        K, _k = len(mylist), 0  # K:base, _k:count
        CP, _cp, _p = len(_color), 0, 0  # CP:base, _cp:pointer, _p:count
        KP = CP + 1  # Constant for color fixpoint
        _x = K * CP  # total number of variations
        for key, value in mylist.items():
            try:
                _p += 1  # horizontal count list Items to length _p = base of CP
                _cp = (_p - 1) % CP  # color position
                x = _p % CP  # horizontal counter color mod CP base round
                y = (x if x > 0 > x else (x + 1)) + (K + (_k - _p))  # vertical counter mylist KV, degrease -y = x - K
                z = abs(_k % y)  # vertical for mod(x) rest z>0,z<0,!=0 for color counter
                
                _base = lambda a, b, n, c: (((a * b) * n - 1) * n) % b - n - c  # c is a const for fixpoint
                _det = lambda a, b, c: (a + (c - (c - (-c))))  # expect -1
                _c = _base(_k, K, _k, 5)
                e = (_k + K + (-1))
                f = (e - (-1))
                d = (e - f)
                _d = _det(d, e, _c)
                _fixpoint = lambda l: (((_d + _c + _k) + (_k - _k - l)) % KP) % l  # expect 0
                _fix = lambda l: (((_d + _c + _k) + (_k - _k - l)) % KP) % l == 0  # expect TRUE
                
                if _k == 0:
                    _newlist = [_color[_cp], key, value]
                    if FlagSwitch.VERBOSE == 5:
                        print(Colors.RESET)
                        print(':BLOCK:DUMMY:ZERO:')  # convention as :[TYPE]:[FUNCTION]:[CONTENT]:
                        print('||', _k, '||', _cp, CP, '|', x, y, z, '||', K, '|', _x)
                        print('||', _k, '||', K, '|', _c, _d, '|', f, e, '||', _fixpoint(KP), _fix(KP))
                        print('||', _k, '||', _newlist, '|Z|')
                else:
                    _newlist = [_color[_cp], key, value]
                
                _filter = _newlist
                sorted.append(_newlist)
                
                for pi, each in enumerate(_color):
                    _k += 1  # horizontal count list Items to length K = basis
                    _c = _base(_k, K, _k, 5)
                    e = (_k + K + (-1))
                    f = (e - (-1))
                    d = (e - f)
                    _d = _det(d, e, _c)
                    
                    _newlist = [each, key, value]
                    
                    if _fix(KP):
                        _filter = _newlist
                    newlisted.append(_filter)
                    
                    if FlagSwitch.VERBOSE == 4:
                        log.info(
                            f'#{round(((_k / 10) / 10) * 10 * 10, 1)} |{each}◾Item:{_newlist}:[{_p}/{_cp}/{CP}]|[{_k}/{K}/{_x}]{Colors.RESET}◾| dict_traversal')
                    if FlagSwitch.VERBOSE == 5:
                        print(each)
                        print('||', _k, '||', _cp, '|', x, y, z, '||')
                        print('||', _k, '|', x, '|', _c, _d, '|', f, e, '||', _fixpoint(KP), _fix(KP))
                        print('||', _k, '||', _newlist, '|C|')  # Current Filter item
            
            except AttributeError as e:
                log.exception(f'{Colors.__name__} | ', e)
            else:
                if isinstance(value, dict):
                    Colors.color_rainbow_dict(mylist)
                # maybe needed, or maybe not
                # if isinstance(value, (list, str)):
                # Colors.recursive_list_traversal(value[-1:])
            finally:
                if FlagSwitch.VERBOSE == 5:
                    log.info(
                        f'Item[{_k}/{K}/{_x}]:{_color[_cp]}◾{_filter[1:]}{Colors.RESET}◾|Color[{_cp}/{CP}]:{_color[_cp]}◽{[_color[_cp][2:]]}◾▁▄▆▓▒░◠◡▸▏◽{Colors.RESET}')
                if FlagSwitch.VERBOSE == 5:
                    print(Colors.RESET)
                    print('||', _k, f'|| S #{_p}', sorted, Colors.FG.LIGHT_GREY)
                if FlagSwitch.EXT == 9:  # disabled, maybe needed, or maybe not
                    print('||', _k, f'|| N #{_k}', newlisted)
        # end:for:loop(mylist.items())
        return sorted


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name
    
class DirectoryTree():
    """
    ## Konfiguration ensure_directories
    """
    
    def __init__(self, build = False):
        self.build = build
    
    
    # [Index - Textabschnitt - Code (z.B. G1-1_01) - Heading - Title - Seite]
    # TODO: structured directory liste für ein build einrichten
    COMPANY = "HannoverRBCck" #,"Commerzbank","Hypoport")
    SHEET_NAME = "ESRS G1"
    API_IMPORT = "API/ImportReport/Nachhaltigkeitsbericht_" + COMPANY + "_" + "results.json"
    API_RESULT = "API/ExportResult/" + "Datapoint-Report_result.json"
    API_SRC_EXPORT = "API/ExportResult/"
    API_TEMP_FILE = "Datapoint-Report_result.json"
    XLS_DATAPOINTS = "API/resources/knowledgebase/datapoints/EFRAG_IG_3_List_of_ESRS.xlsx"
    EXT_DATAPOINTS = "API/resources/datapoints/extracted/ESRS-Datapoints"
    PROMPT_DATAPOINTS = "API/resources/datapoints/prompts/ESRS-Datapoints"
    PROMPT_FINE_TUNINGS = "API/resources/datapoints/prompts/ESRS-Datapoints_prompts.jsonl"
    API_CONVOLUTIONS = "API/resources/datapoints/convoluted/" # + "Datapoint-Report_log"
    
    DIRS = [API_IMPORT, API_RESULT, API_SRC_EXPORT, API_TEMP_FILE, XLS_DATAPOINTS, EXT_DATAPOINTS, PROMPT_DATAPOINTS, PROMPT_FINE_TUNINGS, API_CONVOLUTIONS]
    FILES = [API_SRC_EXPORT, EXT_DATAPOINTS, API_CONVOLUTIONS, API_RESULT]
    INOUT = [XLS_DATAPOINTS, EXT_DATAPOINTS, PROMPT_DATAPOINTS]
    APIPORT = [API_IMPORT, API_RESULT, API_SRC_EXPORT+API_TEMP_FILE]
    
    
    @staticmethod
    def iter_list(lst:list[str]):
        dirs = []
        for i, dir in enumerate(lst, 0):
            dirs.append(dir)
            print(i,dir)
        return dirs
    
    
    @staticmethod
    def ensure_directories(base_path: str = '.', folders: list[str] = ['extracted', 'prompts']) -> dict:
        """
        Erstellt angegebene Verzeichnisse unterhalb des Basispfads, falls sie fehlen.
        Args:
            base_path (str): Wurzelverzeichnis, unter dem die Ordner angelegt werden.
            folders (list): Liste von Ordnern oder Pfaden (auch verschachtelt möglich).
        Returns:
            dict: Dictionary mit den endgültigen Pfadnamen pro Ordner.
        """
        created_paths = {}
        for folder in folders:
            full_path = os.path.join(base_path, folder)
            os.makedirs(full_path, exist_ok=True)
            created_paths[folder] = full_path
        return created_paths
        
    @staticmethod
    def ensure_directories(base_path: str = '.', dirs: list[str] = ['extracted', 'prompts']):
        for d in dirs:
            full_path = os.path.join(base_path, d)
            if os.path.exists(full_path):
                if not os.path.isdir(full_path):
                    log.error(f"Path exists and is not a directory: {full_path}")
                    continue
            else:
                os.makedirs(full_path, exist_ok=True)



class LogFlags(IntFlag):
    """
    # Example usage:
    log_flags = LogFlags.INFO_LOG | LogFlags.VERBOSE | LogFlags.GREEN
    print(log_flags)
    """
    
    #################
    # print(log.getLevelNamesMapping())
    # {'CRITICAL': 50, 'FATAL': 50, 'ERROR': 40, 'WARN': 30, 'WARNING': 30, 'INFO': 20, 'DEBUG': 10, 'NOTSET': 0, 'FRAME': 70, 'KEYPUT': 80, 'FLAGS': 90}
    # self.log_lvl = log.getLevelName('FRAME')
    # log.basicConfig(level=lg.DEBUG, format='%(asctime)-15s %(name)-5s %(levelname)-8s %(message)s')
    ################
    
    INFO_LOG = 8
    LOG_FILE = 16
    
    YELLOW = 32
    GREEN = 64
    BLUE = 128
    PURPLE = 256
    
    DEBUG = 10
    FRAME = 70
    KEYPUT = 80
    FLAGS = 90
    
    LOGFLAGS = INFO_LOG & LOG_FILE
    LVL_MODES = DEBUG & FRAME & KEYPUT & FLAGS
    COLORS = YELLOW | GREEN | BLUE | PURPLE
    
    VERBOSE_COLOR = COLORS
    VERBOSE_HIGHLIGHT = COLORS
    VERBOSE_HIGHLIGHT_2 = COLORS
    VERBOSE_HIGHLIGHT_3 = COLORS
    
    def __str__(self):
        return f'{self.LOGFLAGS, self.LVL_MODES, self.COLORS}'
    
    def __add__(self, other):
        return other
    