# pin= "881120-1068234"
# yyyymmdd= int(pin[:6]) + 19000000
# num= int(pin[-7:])
#
# print(yyyymmdd, num)
# print('Male' if num//1000000 == 1 else 'Female')
#
# a= [1,3,5,4,2]
# a.sort()        #No return_ Nonetype
# a_rev= sorted(a, reverse= True)
# print(a, a_rev)
# # print(a.pop(0))
#
# quote= ['life', 'is', 'too', 'short']
# join_quote= " ".join(quote)
# print(join_quote.split(" "))
#
# b= (1,2,3)
# b= b+ (4,5)     #Tuple은 수정 불가능하지만,
#                 #예외적으로 +, *은 파이썬 내장함수로 새로운 튜플 데이터를 만든다.
#                 #그리고 a가 새로운 튜플을 참조하므로 b가 변화하는 것처럼 보이게 됨.
#                 #뺄셈 등은 set에서만 가능하며, tuple에서는 del도 사용 불가능하다.
#
# print(b)
#
# c= {'a': 90, 'b': 80, 'c': 70}
# result= c.pop('b')
# print(result, c)
#
# a= [1,1,1,2,2,3,3,3,4,4,5]
# a_set= set(a)
# b_set= {2*x+1 for x in range(5)}
# u_set= set(range(10))
# print(u_set-(a_set & b_set), (u_set - a_set)|(u_set - b_set))
# print(u_set-(a_set & b_set) == (u_set - a_set)|(u_set - b_set))
#
# a= "life is too short, you need python"
# print(list(a))
#
# if 'or' in a: print('or')
# else: print('none')       # in 함수에서 a에 있는 단어별로 슬라이싱이 된다?
#
# i= 0
# while True:
#     i += 1
#     if i>5: break
#     print('*'*i)
#
a= [1,2,3,4,5]
# for i in a:
#     a.append(i)
#     print(a)
#     if len(a)== 50: break

a= [70, 60, 55, 75, 95, 90, 80, 80, 85, 100]
print(sum(a)/len(a))
print(sum(range(10)))
