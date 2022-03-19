from base64 import b64decode, b64encode


with open("inputb64",'rb') as file:
    a = b64decode(file.read())

    with open("test.jpg",'wb') as out:
        out.write(a)
