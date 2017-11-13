import matplotlib.pyplot as plt


# 밑바닥 3장 p.46 예시 data
variance = [1, 2, 4, 8, 16, 32, 64, 128, 256]
bias_squared = [256, 128, 64, 32, 16, 8, 4, 2, 1]
total_error = [x+y for x, y in zip(variance, bias_squared)]
xs = [i for i,_ in enumerate(variance)]

# 꾸미기 plot(x, y, color-linestyle-marker) 표현방식 다양, 추가 가능한 속성도 다양
plt.plot(xs, variance, color='green', linestyle='--')
plt.plot(xs, variance, 'g--', label = 'variance')
plt.plot(xs, bias_squared, 'r-', label = 'bias^2')
plt.plot(xs, total_error, 'b:', label = 'total error')
plt.legend(loc="top center")

plt.title("The Bias-variance Tradeoff")
plt.xlabel("model complexity")

plt.show()


#plot (x, y, color="red", linestyle="--", marker="o" ) -> plot (x, y, r--o)

#
# # 그래프 중첩(한 chart 위에 여러개 series) (multiple lines)
# plt.plot([1,2,3], [4,5,6], 'r--')
# plt.plot([4,5,6], [8,10,12], 'b*', )
# plt.legend()
# plt.show()
#
#
# # annotate
# plt.plot(X, C, label="cosine")
# t = 2 * np.pi / 3
# plt.scatter(t, np.cos(t), 50, color='blue')
# plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$', xy=(t, np.cos(t)), xycoords='data', xytext=(-90, -50),
#              textcoords='offset points', fontsize=16, arrowprops=dict(arrowstyle="->"))
#
# # multiplot