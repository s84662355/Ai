package main

import (
    "fmt"
    "image/color"
    "math/rand"
    "time"

    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
)

func main() {
    // 设置随机数种子
    rand.Seed(time.Now().UnixNano())

    // 数据参数
    const (
        n        = 1000    // 数据点数量
        xMin     = 0.0    // x最小值
        xMax     = 10.0   // x最大值
        noiseMax = 3.554646    // 最大噪声值
    )

    // 真实模型参数 y = 1.342x + 2.45
    slope := 1.342
    intercept := 2.45

    // 生成数据
    x := make([]float64, n)
    y := make([]float64, n)
    // 用于存储标识，1 表示在直线上方，0 表示在直线下方
    labels := make([]int, n) 
    for i := 0; i < n; i++ {
        x[i] = xMin + rand.Float64()*(xMax-xMin)                  // x均匀分布
        noise := (rand.Float64()*2 - 1) * noiseMax                 // 噪声范围: [-1.5, 1.5]
        y[i] = slope*x[i] + intercept + noise                      // 带噪声的y值
        
        // 计算真实直线在该 x 处的 y 值
        trueY := slope * x[i] + intercept
        if y[i] > trueY {
            labels[i] = 1
        } else {
            labels[i] = 0
        }
    }

    // 1. 修正 plot.New() 返回值：只返回1个对象和1个错误
    p  := plot.New()
 
    // 坐标系标题和轴标签
    p.Title.Text = "平面坐标系下的数据分布 (y = 1.342x + 2.45)"
    p.X.Label.Text = "X轴"
    p.Y.Label.Text = "Y轴"

    // 设置坐标轴范围
    p.X.Min = xMin - 1
    p.X.Max = xMax + 1
    p.Y.Min = (slope*xMin + intercept) - noiseMax - 1
    p.Y.Max = (slope*xMax + intercept) + noiseMax + 1

    // 2. 修正网格线：部分版本 Grid 无 Color 字段，移除颜色设置
    grid := plotter.NewGrid()
    p.Add(grid) // 直接添加网格，使用默认样式

    // 绘制数据点（散点图）
    scatterData := make(plotter.XYs, n)
    for i := range x {
        scatterData[i] = plotter.XY{X: x[i], Y: y[i]}
    }
    scatter, err := plotter.NewScatter(scatterData)
    if err != nil {
        panic(err)
    }
    scatter.Color = color.RGBA{R: 255, G: 0, B: 0, A: 255} // 红色数据点
    scatter.Radius = vg.Points(3)                          // 点大小
    p.Add(scatter)
    p.Legend.Add("数据点", scatter)

    // 绘制真实直线
    lineData := make(plotter.XYs, 2)
    lineData[0] = plotter.XY{X: xMin, Y: slope*xMin + intercept}
    lineData[1] = plotter.XY{X: xMax, Y: slope*xMax + intercept}
    line, err := plotter.NewLine(lineData)
    if err != nil {
        panic(err)
    }
    line.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255} // 蓝色直线
    line.Width = vg.Points(2)                          // 线粗细
    p.Add(line)
    p.Legend.Add("真实直线 y=1.342x+2.45", line)

    p.Legend.Top = true // 图例放在顶部

    // 保存图表
    outputFile := "linear_coordinate_system.png"
    if err := p.Save(10*vg.Inch, 8*vg.Inch, outputFile); err != nil {
        panic(err)
    }
    fmt.Printf("平面坐标系图表已保存为 %s\n", outputFile)

    // 打印部分数据及标识
    fmt.Println("\n前5个数据点的坐标及标识（1：直线上方，0：直线下方）：")
    for i := 0; i < 5; i++ {
        fmt.Printf("(x=%.2f, y=%.2f) 标识: %d\n", x[i], y[i], labels[i])
    }
}