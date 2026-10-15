n,a,b = 0,0,1
while n < 1000:
    print(b)
    a, b = b,a+b
    n += 1
print("Done")