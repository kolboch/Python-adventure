import mainFile


def main():
    try:
        file = None
        file = mainFile.getFile('text.txt')
    except IOError:
        print('Catched IOError when trying to open file')
    finally:
        if(file):
            mainFile.closeFile(file)
    print(dir(mainFile))

if __name__ == '__main__':
    main()