
def callback(mydict={"a":"A", "b":"B", "c":"C"}):
    for name in mydict:
        print(name)
        print(mydict[name])

newdict = {"a":"1", "b":"2"}
callback(newdict)