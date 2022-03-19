from base64 import b64encode


with open("cat.jpg",'rb') as file:
    s = b64encode(file.read())
    with open("inputb64",'wb') as out:
        out.write(s)

