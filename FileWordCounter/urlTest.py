import urllib.request
import re

def readfileFromUrl(url):
    response = urllib.request.urlopen(url)
    output = response.read().decode(response.headers.get_content_charset())
    m = re.match('.*\.png', output)
    if m:
        print(m.group())
    else:
        print('not found')

def retrieveUrl(url, fileNameToSave):
    urllib.request.urlretrieve(url, fileNameToSave)

def main():
    retrieveUrl('https://upload.wikimedia.org/wikipedia/commons/thumb/c/c2/Faust_bei_der_Arbeit.JPG/220px-Faust_bei_der_Arbeit.JPG','test.jpg')

if __name__ == '__main__':
    main()