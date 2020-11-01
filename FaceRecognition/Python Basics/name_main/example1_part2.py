import eaxmple1_part1
def main():
    print("Welcome to main function of part 2")
if __name__ == '__main__':
    print("Part 2: Python is running this file directly so name is __main__")
    print("Second Module's name is " + __name__)
    eaxmple1_part1.main()
    main()
else:
    print("Part 2: Python is not running this file directly so name is changed to file name with which it is made")
    print("First Module's name is " + __name__)