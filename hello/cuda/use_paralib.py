from ctypes import windll

para = windll.LoadLibrary("paralib/build/Debug/paralib.dll")

para.printHello()