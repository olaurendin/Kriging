l = [i for i in range(10)]

def yieldy_function(l):
    for i in l:
        yield i

l2 = yieldy_function(l)
print(l, l2)
for i in l2:
    print(l2)
