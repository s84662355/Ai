package main

import (
	"fmt"

	"gonum.org/v1/gonum/stat/distuv"
)

func main() {
	// 模拟样本数据，实际使用时替换为真实数据
	data := sampleddata(50000)
	lr := 0.01            // 学习率
	initialB := 0.0       // 初始化 b 为 0
	initialW := 0.0       // 初始化 w 为 0
	numIterations := 1000 // 迭代次数

	// 执行梯度下降，优化 w 和 b
	b, w := GradientDescent(data, initialB, initialW, lr, numIterations)

	// 计算最终的均方误差
	loss := Mse(b, w, data)

	// 打印最终结果
	fmt.Printf("Final loss:%f, w:%f, b:%f\n", loss, w, b)
}

func sampleddata(numSamples int) [][]float64 {
	// 用于保存样本数据，每个元素是一个包含两个 float64 元素的切片（类似 Python 中的 [x, y] ）
	data := make([][]float64, 0, numSamples)

	for i := 0; i < numSamples; i++ {
		// 随机采样输入 x，范围在 -10 到 10 之间，类似 Python 中的 np.random.uniform(-10., 10.)
		uniformDist := distuv.Uniform{Min: -10.0, Max: 10.0, Src: nil}
		x := uniformDist.Rand()

		// 采样高斯噪声，均值为 0.，标准差为 0.01，类似 Python 中的 np.random.normal(0., 0.01)
		normalDist := distuv.Normal{Mu: 0.0, Sigma: 0.01, Src: nil}
		eps := normalDist.Rand()

		// 得到模型的输出，这里是简单的线性关系 y = 1.477 * x + 0.089 + eps
		y := 1.477*x + 0.089 + eps

		// 保存样本点
		data = append(data, []float64{x, y})
	}

	return data
}

// Mse 计算均方误差，参数 b 是偏置，w 是权重，points 是样本数据，每个元素是 [x, y] 形式的切片
func Mse(b, w float64, points [][]float64) float64 {
	totalError := 0.0
	// 遍历所有样本点
	for _, point := range points {
		x := point[0]
		y := point[1]
		// 计算预测值与真实值的差的平方并累加
		totalError += (y - (w*x + b)) * (y - (w*x + b))
	}
	// 求平均得到均方误差
	return totalError / float64(len(points))
}

// StepGradient 实现梯度下降的一步，更新 b 和 w 的值
// bCurrent: 当前的偏置值
// wCurrent: 当前的权重值
// points: 样本数据，每个元素是 [x, y] 形式的切片
// lr: 学习率
func StepGradient(bCurrent, wCurrent float64, points [][]float64, lr float64) (float64, float64) {
	bGradient := 0.0
	wGradient := 0.0
	M := float64(len(points)) // 总样本数

	for _, point := range points {
		x := point[0]
		y := point[1]

		// 计算误差函数对 b 的梯度
		bGradient += (2 / M) * ((wCurrent*x + bCurrent) - y)
		// 计算误差函数对 w 的梯度
		wGradient += (2 / M) * x * ((wCurrent*x + bCurrent) - y)
	}

	// 根据梯度下降算法更新 b 和 w
	newB := bCurrent - lr*bGradient
	newW := wCurrent - lr*wGradient

	return newB, newW
}

// GradientDescent 实现梯度下降迭代过程，更新 b 和 w 的值
// points: 样本数据，每个元素是 [x, y] 形式的切片
// startingB: b 的初始值
// startingW: w 的初始值
// lr: 学习率
// numIterations: 迭代次数
func GradientDescent(points [][]float64, startingB, startingW, lr float64, numIterations int) (float64, float64) {
	b := startingB
	w := startingW

	for step := 0; step < numIterations; step++ {
		// 调用 StepGradient 计算梯度并更新一次 b 和 w
		b, w = StepGradient(b, w, points, lr)

		// 计算当前的均方误差，用于监控训练进度
		loss := Mse(b, w, points)

		// 每 50 次迭代打印一次误差和实时的 w、b 值（对应 Python 里 step % 50 == 0 的逻辑）
		if step%50 == 0 {
			fmt.Printf("Iteration:%d, loss:%f, w:%f, b:%f\n", step, loss, w, b)
		}
	}

	return b, w
}
