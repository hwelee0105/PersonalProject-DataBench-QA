from operator import index


a = ['1', '2', '3']

print(str(a).toString(index=False))
print(','.join([str(i) for i in a]))
