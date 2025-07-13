package main

import (
    "fmt"
    "image/color"
    "math"
    "math/rand"
    "time"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
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

// 生成数据，包括x数组、y数组、标签数组（1表示在直线上方，0表示在直线下方）
func generateData(n int, slope, intercept, noiseMax float64) (
    []float64, []float64, []int) {
    rand.Seed(time.Now().UnixNano())
    x := make([]float64, n)
    y := make([]float64, n)
    labels := make([]int, n)
    xMin := 0.0
    xMax := 10.0
    for i := 0; i < n; i++ {
        x[i] = xMin + rand.Float64()*(xMax-xMin)
        noise := (rand.Float64()*3 - 1) * noiseMax
        y[i] = slope*x[i] + intercept + noise
        trueY := slope*x[i] + intercept
        if y[i] > trueY {
            labels[i] = 1
        } else {
            labels[i] = 0
        }
    }
    return x, y, labels
}

// 绘制图像，包括原始数据、真实直线、训练后模型拟合的直线（这里简单用最终参数绘制近似直线示意）
func plotData(xData, yData []float64, labels []int, 
    trueSlope, trueIntercept, a, b, c float64) {
    p := plot.New()
 
    p.Title.Text = "Data Distribution and Fitted Line"
    p.X.Label.Text = "X"
    p.Y.Label.Text = "Y"

    // 绘制真实直线
    trueLineData := make(plotter.XYs, 2)
    trueLineData[0] = plotter.XY{X: 0, Y: trueIntercept}
    trueLineData[1] = plotter.XY{X: 10, Y: trueSlope*10 + trueIntercept}
    trueLine, err := plotter.NewLine(trueLineData)
    if err != nil {
        panic(err)
    }
    trueLine.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}
    trueLine.Width = vg.Points(2)
    p.Add(trueLine)
    p.Legend.Add("True Line (y=1.342x+2.45)", trueLine)

    // 绘制数据点，区分上下方（这里简单用颜色区分，1用红色，0用蓝色）
    scatterUp := make(plotter.XYs, 0)
    scatterDown := make(plotter.XYs, 0)
    for i := range xData {
        if labels[i] == 1 {
            scatterUp = append(scatterUp, plotter.XY{X: xData[i], Y: yData[i]})
        } else {
            scatterDown = append(scatterDown, plotter.XY{X: xData[i], Y: yData[i]})
        }
    }
    // 上方点
    upPlotter, err := plotter.NewScatter(scatterUp)
    if err != nil {
        panic(err)
    }
    upPlotter.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255}
    upPlotter.Radius = vg.Points(3)
    p.Add(upPlotter)
    p.Legend.Add("Above Line (1)", upPlotter)
    // 下方点
    downPlotter, err := plotter.NewScatter(scatterDown)
    if err != nil {
        panic(err)
    }
    downPlotter.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}
    downPlotter.Radius = vg.Points(3)
    p.Add(downPlotter)
    p.Legend.Add("Below Line (0)", downPlotter)

    // 绘制拟合的直线（这里用 logistic 函数的线性部分近似，简单绘制示意，实际是分类边界）
    // 拟合的线性关系是 g(x,y)=a + b*x + c*y，这里为了可视化，固定 y 或者做简化，假设 y 取数据中的值等，
    // 这里简单绘制当 g(x,y)=0 时的直线近似（分类边界），实际逻辑回归是二分类，边界是 g(x,y)=0 即 a + b*x + c*y = 0
    // 为了在 2D 图像展示，假设 y 是另一个维度，这里简化处理，比如取 y 为数据中的均值等，或者固定一个值，这里简单演示
    // 以下只是示意，实际根据你的模型理解调整可视化方式，比如如果是多元逻辑回归，可视化会复杂些，这里因为数据生成是基于 y = 1.342x + 2.45 + noise，
    // 可以近似认为拟合的是类似的线性关系，简单绘制 a + b*x + c*y = 0 的直线（分类边界）
    fitLineData := make(plotter.XYs, 2)
    // 找两个点绘制直线，这里简单取 x 范围端点，计算对应的 y
    x1, x2 := 0.0, 10.0
    // 假设 y 是数据集中的 y 维度，这里为了绘图，假设我们想展示在原始数据的 y 维度上的边界，可能需要调整，
    // 以下计算是 a + b*x + c*y = 0 → y = (-a - b*x)/c （c≠0时）
    if c != 0 {
        y1 := (-a - b*x1) / c
        y2 := (-a - b*x2) / c
        fitLineData[0] = plotter.XY{X: x1, Y: y1}
        fitLineData[1] = plotter.XY{X: x2, Y: y2}
        fitLine, err := plotter.NewLine(fitLineData)
        if err != nil {
            panic(err)
        }
        fitLine.Color = color.RGBA{R: 0, G: 255, B: 0, A: 255}
        fitLine.Width = vg.Points(2)
        p.Add(fitLine)
        p.Legend.Add("Fitted Line", fitLine)
    }

    if err := p.Save(10*vg.Inch, 8*vg.Inch, "result_plot.png"); err != nil {
        panic(err)
    }
    fmt.Println("图像已保存为 result_plot.png")
}

func main() {
    // 数据生成参数
    const (
        n        = 2000   // 数据点数量
        slope    = 1.342  // 真实直线斜率
        intercept = 2.45  // 真实直线截距
        noiseMax = 3.554646 // 噪声最大值
    )

    // 生成数据
    xData, yData, yTrue := generateData(n, slope, intercept, noiseMax)

    // 模型初始参数
    a, b, c := 0.0, 0.0, 0.0
    learningRate := 0.006
    iterations := 200000

    // 迭代训练
    for i := 0; i < iterations; i++ {
        gradA := gradientA(yTrue, xData, yData, a, b, c)
        gradB := gradientB(yTrue, xData, yData, a, b, c)
        gradC := gradientC(yTrue, xData, yData, a, b, c)

        a -= learningRate * gradA
        b -= learningRate * gradB
        c -= learningRate * gradC

        // 可选：每轮打印损失，观察收敛情况
        if i%10 == 0 {
            loss := crossEntropyLoss(yTrue, xData, yData, a, b, c)
            fmt.Printf("迭代 %d 次，损失: J=%.4f\n", i, loss)
        }
    }

    // 输出最终参数
    fmt.Printf("训练后参数：a=%.4f, b=%.4f, c=%.4f\n", a, b, c)

    // 绘制图像
    plotData(xData, yData, yTrue, slope, intercept, a, b, c)
}