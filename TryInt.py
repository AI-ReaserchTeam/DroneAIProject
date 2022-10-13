from tkinter import messagebox as msg
from UpperStrip import UpperStrip


def TryInt(Num, String):
    if Num or Num == 0:
        Num = UpperStrip(str(Num))
    try:
        Num = float(Num)
        Num = round(Num)
        return int(Num)
    except ValueError:
        print(str(String), ' must be a numeric integer')
        #msg.showerror('Error', str(String) + ' must be a numeric integer')      # this should be commentted
        return None
    except TypeError:
        print(str(String), ' Should not be empty')
        #msg.showerror('Error', str(String) + '  Should not be empty')
        return None

