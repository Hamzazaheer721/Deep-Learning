
print("This will always run so try to put all this text in main method")
def main():
    print("welcome to main function of part 1")
    print("The First Module's name is " , __name__)
if __name__ == '__main__':
    print("Part 1: Python is running this file directly so name is __main__")
    main()
else:
    print("Part 1: Python is not running this file directly so name is changed to file name with which it is made")