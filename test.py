def gener(N):
    for i in range(N):
        yield i
    return 25125152

res = [x for x in gener(5)]
print(res)