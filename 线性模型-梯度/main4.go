package main

import (
    "fmt"
    "math"
)

// 计算逻辑斯蒂函数：P(x) = 1 / (1 + e^(-g(x)))
func logistic(a, b, c, x, y float64) float64 {
    g := a + b*x + c*y
    return 1.0 / (1.0 + math.Exp(-g))
}

// 计算平均交叉熵损失
func crossEntropyLoss(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
    m := len(yTrue)
    sumLoss := 0.0
    for i := range yTrue {
        p := logistic(a, b, c, xData[i], yData[i])
        if yTrue[i] == 1 {
            sumLoss -= math.Log(p)
        } else {
            sumLoss -= math.Log(1 - p)
        }
    }
    return sumLoss / float64(m)
}

// 计算损失函数对 a 的梯度
func gradientA(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
    m := len(yTrue)
    sum := 0.0
    for i := range yTrue {
        p := logistic(a, b, c, xData[i], yData[i])
        sum += (p - float64(yTrue[i]))
    }
    return sum / float64(m)
}

// 计算损失函数对 b 的梯度
func gradientB(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
    m := len(yTrue)
    sum := 0.0
    for i := range yTrue {
        p := logistic(a, b, c, xData[i], yData[i])
        sum += (p - float64(yTrue[i])) * xData[i]
    }
    return sum / float64(m)
}

// 计算损失函数对 c 的梯度
func gradientC(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
    m := len(yTrue)
    sum := 0.0
    for i := range yTrue {
        p := logistic(a, b, c, xData[i], yData[i])
        sum += (p - float64(yTrue[i])) * yData[i]
    }
    return sum / float64(m)
}

func main() {
    // 示例数据（3 个样本）
    yTrue := []int{1, 0, 1}
    xData := []float64{1.0, 2.0, 3.0}
    yData := []float64{1.0, 2.0, 3.0}
    a, b, c := 0.0, 0.0, 0.0 // 初始参数

    // 计算初始损失
    loss := crossEntropyLoss(yTrue, xData, yData, a, b, c)
    fmt.Printf("初始损失: J=%.4f\n", loss)

    // 计算梯度
    gradA := gradientA(yTrue, xData, yData, a, b, c)
    gradB := gradientB(yTrue, xData, yData, a, b, c)
    gradC := gradientC(yTrue, xData, yData, a, b, c)

    fmt.Printf("对 a 的梯度: dJ/da=%.4f\n", gradA)
    fmt.Printf("对 b 的梯度: dJ/db=%.4f\n", gradB)
    fmt.Printf("对 c 的梯度: dJ/dc=%.4f\n", gradC)
}