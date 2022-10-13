#   from check import check
#   cd "Documents\DroneAIProject"  pyinstaller -F --add-data C:\Python39\tcl\tix8.4.3;tcl\tix8.4.3 MainEntry.py
#   python3.9 -m pyinstaller
#   Kidnap_Banditry_Bandits_Robbery_Normal



def UpperStrip(code):
    #   check()
    c = code.upper()
    c = c.strip()
    c = c.replace(' ', '')
    return c

