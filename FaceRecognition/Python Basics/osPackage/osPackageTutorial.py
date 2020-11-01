import os
from datetime import datetime
# print (dir(os))
# print( os.getcwd())

# print(os.chdir('C:/'))
# print(os.getcwd())
# print(os.listdir()) # this will print directories of D again
# for printing the directories of C you gotta include previous chdir commands

# os.mkdir("Thonnymkdri")
# os.rmdir("Thonnymkdri")
# print(os.listdir())

# os.makedirs("thonnymakedirs/dir1/dir2")
# os.removedirs("thonnymakedirs/dir1/dir2")
# os.chdir("thonnymakedirs")
# os.rename("thonnymakedirs","fileRenamed")
# os.removedirs("fileRenamed/dir1/dir2")
# print(os.listdir())
# print(os.getcwd())
# modtime = os.stat('osReview.py').st_mtime
# print(datetime.fromtimestamp(modtime))

# Base_path = os.getcwd()     # tells the path where our running program is running but wont mention running folder name
# b = os.path.basename(os.path.abspath(__file__))  # only tells the running file name in abspath
# print(os.path.abspath(__file__))               # it will tell whole abspath including running file name
# print(os.path.dirname(__file__))        # it is same as  getcwd() but it can be used for fake paths unlike getcwd
# print(img_dir)
# print(os.path.basename(os.path.dirname(__file__)))  # base name just gives most last traversed directory to be shown
# print(b)




# for dirpath, dirnames, filenames in os.walk("C:/Users/Ricardo-PC/Desktop"):
#     print("Current Path", dirpath)
#     print("Directories in Current path", dirnames)
#     print("Files in Current Path", filenames)
#     print()

# lets try getting environment variables
# print(os.environ.get("HOMEDRIVE"))

# creating a new path from os.environ
# print(os.environ)
new_file_path = os.path.join(os.environ.get("APPDATA"), 'ali.txt')
print(new_file_path)
# 'w' is used if the file is being created for first time and has not existed earlier.
# 'w' being used again on file will overwrite
# 'a' will append
# 'r' is used for readin while opening file
with open(new_file_path, 'w') as f:
    f.write("hello my friend \n")
    print()
f = open(new_file_path, 'r')
print(f.read())
print(os.path.exists(new_file_path))  # true
print(os.path.isdir(new_file_path))   # false
print(os.path.isfile(new_file_path))   # true
print(os.path.splitext(new_file_path))   # 'C:\\Users\\Ricardo-PC\\AppData\\Roaming\\ali', '.txt')
print(os.path.dirname(new_file_path))  # C:\Users\Ricardo-PC\AppData\Roaming
print(os.path.basename(new_file_path))  # ali.txt
print(os.path.split(new_file_path))  # ('C:\\Users\\Ricardo-PC\\AppData\\Roaming', 'ali.txt')
print(os.path.basename("/dir/lingo.py"))  # lingo.py
print(os.path.dirname("/ali/dir/lingo.py"))  # /ali/dir
print(os.path.exists("ok/hello.py"))  # false
ok = (os.path.splitext("ok/hello.py"))  # ('ok/hello', '.py')
print(ok)
print(ok[1])
print(os.path)
print(dir(os.path))