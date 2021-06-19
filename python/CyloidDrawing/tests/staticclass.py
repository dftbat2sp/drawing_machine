class foo:

    the_biggest_num=0

    def __init__(self, mynum):
        self.num=mynum
        foo.the_biggest_num=max(mynum,foo.the_biggest_num)

one=foo(2)
two=foo(3)
three=foo(1)

print(foo.the_biggest_num)