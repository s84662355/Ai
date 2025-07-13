package main

import (
	"fmt"
	//"image/color"
	//"math"
	"math/rand"
	"time"

	"github.com/pa-m/sklearn/linear_model"
	"gonum.org/v1/gonum/mat"
	//"gonum.org/v1/plot"
	//"gonum.org/v1/plot/plotter"
	//"gonum.org/v1/plot/vg"
)

// 生成数据：x数组、y数组、标签（1=上方，0=下方）
func generateData(n int, slope, intercept, noiseMax float64) ([]float64, []float64, []int) {
	rand.Seed(time.Now().UnixNano())
	x := make([]float64, n)
	y := make([]float64, n)
	labels := make([]int, n)
	xMin, xMax := 0.0, 10.0
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

// 转换数据为模型所需的矩阵格式
func prepareData(xData, yData []float64) *mat.Dense {
	n := len(xData)
	data := make([]float64, 0, n*2) // 每行2个特征：[x, y]
	for i := 0; i < n; i++ {
		data = append(data, xData[i], yData[i])
	}
	return mat.NewDense(n, 2, data) // n行2列矩阵
}

// // 绘制图像
// func plotData(xData, yData []float64, labels []int, trueSlope, trueIntercept float64, model *linearmodel.LogisticRegression) {
// 	p  := plot.New()
 
// 	p.Title.Text = "Data Distribution and Fitted Line (sklearn)"
// 	p.X.Label.Text = "X"
// 	p.Y.Label.Text = "Y"

// 	// 绘制真实直线
// 	trueLineData := plotter.XYs{
// 		{X: 0, Y: trueIntercept},
// 		{X: 10, Y: trueSlope*10 + trueIntercept},
// 	}
// 	trueLine, err := plotter.NewLine(trueLineData)
// 	if err != nil {
// 		panic(err)
// 	}
// 	trueLine.Color = color.RGBA{B: 255, A: 255} // 蓝色
// 	trueLine.Width = vg.Points(2)
// 	p.Add(trueLine)
// 	p.Legend.Add("True Line (y=1.342x+2.45)", trueLine)

// 	// 绘制数据点（区分上下方）
// 	scatterUp := make(plotter.XYs, 0)
// 	scatterDown := make(plotter.XYs, 0)
// 	for i := range xData {
// 		if labels[i] == 1 {
// 			scatterUp = append(scatterUp, plotter.XY{X: xData[i], Y: yData[i]})
// 		} else {
// 			scatterDown = append(scatterDown, plotter.XY{X: xData[i], Y: yData[i]})
// 		}
// 	}

// 	// 上方点（红色）
// 	upPlotter, err := plotter.NewScatter(scatterUp)
// 	if err != nil {
// 		panic(err)
// 	}
// 	upPlotter.Color = color.RGBA{R: 255, A: 255}
// 	upPlotter.Radius = vg.Points(3)
// 	p.Add(upPlotter)
// 	p.Legend.Add("Above Line (1)", upPlotter)

// 	// 下方点（蓝色）
// 	downPlotter, err := plotter.NewScatter(scatterDown)
// 	if err != nil {
// 		panic(err)
// 	}
// 	downPlotter.Color = color.RGBA{B: 255, A: 255}
// 	downPlotter.Radius = vg.Points(3)
// 	p.Add(downPlotter)
// 	p.Legend.Add("Below Line (0)", downPlotter)

// 	// 绘制模型的决策边界（分类线）
// 	// 决策边界公式：model.Coef[0]*x + model.Coef[1]*y + model.Intercept[0] = 0
// 	// 转换为 y = (-model.Intercept[0] - model.Coef[0]*x) / model.Coef[1]
// 	coef := model.Coef
// 	intercept := model.Intercept
// 	if len(coef) == 2 && math.Abs(coef[1]) > 1e-10 {
// 		fitLineData := plotter.XYs{
// 			{X: 0, Y: (-intercept[0] - coef[0]*0) / coef[1]},
// 			{X: 10, Y: (-intercept[0] - coef[0]*10) / coef[1]},
// 		}
// 		fitLine, err := plotter.NewLine(fitLineData)
// 		if err != nil {
// 			panic(err)
// 		}
// 		fitLine.Color = color.RGBA{G: 255, A: 255} // 绿色
// 		fitLine.Width = vg.Points(2)
// 		p.Add(fitLine)
// 		p.Legend.Add("Fitted Line (sklearn)", fitLine)
// 	}

// 	// 保存图像
// 	if err := p.Save(10*vg.Inch, 8*vg.Inch, "sklearn_result_plot.png"); err != nil {
// 		panic(err)
// 	}
// 	fmt.Println("图像已保存为 sklearn_result_plot.png")
// }

func main() {
	// 数据参数
	const (
		n         = 2000
		slope     = 1.342
		intercept = 2.45
		noiseMax  = 3.554646
	)

	// 生成数据
	xData, yData, labels := generateData(n, slope, intercept, noiseMax)

	// 准备特征矩阵（每行2个特征：x和y）
	X := prepareData(xData, yData)

	// 转换标签为float64类型（模型要求）
	y := make([]float64, n)
	for i, label := range labels {
		y[i] = float64(label)
	}
	yMat := mat.NewDense(n, 1, y) // 转换为列矩阵

	// 创建逻辑回归模型（根据文档：NewLogisticRegression返回*LogisticRegression）
	model := linearmodel.NewLogisticRegression()
	model.Alpha = 1e-5

	// 配置模型参数（文档说明：通过结构体字段直接设置）
	model.MaxIter = 200000 // 最大迭代次数
	//model.Tol = 1e-8       // 收敛容差（可选）

	// 训练模型（文档说明：Fit方法接收特征矩阵X和标签矩阵y）
	if err := model.Fit(X, yMat); err != nil {
		panic(err)
	}

	// 输出模型参数（文档说明：模型参数存储在Coef和Intercept字段）
	fmt.Println("sklearn逻辑回归模型参数：")
	fmt.Printf("偏置项 a = %.4f\n", model.Intercept[0])   // 对应 a
	fmt.Printf("特征系数  %.4f \n",  model.Coef ) // 对应 b 和 c

	// 绘制图像
	//plotData(xData, yData, labels, slope, intercept, model)
}