package main

import (
	"flag"
	"fmt"
	 
	"log"
	    "image/color" 
 
    "gonum.org/v1/plot"
    "gonum.org/v1/plot/plotter"
    "gonum.org/v1/plot/vg"
 
	 
"github.com/pa-m/sklearn/linear_model"
	"github.com/pa-m/sklearn/base"
	//"github.com/pa-m/sklearn/datasets"
 
	"gonum.org/v1/gonum/mat"
	 
 	"math/rand"
	"time"
)

var _ base.Predicter = &linearmodel.LogisticRegression{}
var visualDebug = flag.Bool("visual", false, "output images for benchmarks and test data")
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
func main() {
	// 数据参数
	const (
		n         = 2000
		slope     = 1.342
		intercept = 2.45
		noiseMax  = 1.554646
	)

	// 生成数据
	xData, yData, labels := generateData(n, slope, intercept, noiseMax)

	// 准备特征矩阵（每行2个特征：x和y）
	X := prepareData(xData, yData)

//	fmt.Println(xData, yData)

	// 转换标签为float64类型（模型要求）
	y := make([]float64, n)
	for i, label := range labels {
		y[i] = float64(label)
	}
	yMat := mat.NewDense(n, 1, y) // 转换为列矩阵

	// adapted from http://scikit-learn.org/stable/_downloads/plot_iris_logistic.ipynb
	//ds := datasets.LoadIris()

	// we only take the first _ features.
	//nSamples, _ := ds.X.Dims()
	//X, YTrueClasses := ds.X.Slice(0, nSamples, 0, 2).(*mat.Dense), ds.Y
 

	regr := linearmodel.NewLogisticRegression()
	regr.Alpha = 1e-5
	regr.Tol = 0.0032

	regr.MaxIter  = 10000
	regr.NIterNoChange = 10
 

	log.SetPrefix("ExampleLogisticRegression_Fit_iris:")
	defer log.SetPrefix("")

	// we create an instance of our Classifier and fit the data.
	regr.Fit(X, yMat)
	//regr.Fit(X, YTrueClasses)

	accuracy := regr.Score(X, yMat)
	if accuracy >= 0.833 {
		fmt.Println("ok")
	} else {
		fmt.Printf("Accuracy:%.3f\n", accuracy)
	}

	plotData(xData, yData,labels, slope, intercept, regr.Intercept[0], regr.Coef.Data       [0], regr.Coef .Data       [1])


	fmt.Println("sklearn逻辑回归模型参数：")
	fmt.Printf("偏置项 a = %.4f\n", regr.Intercept[0])   // 对应 a
	fmt.Printf("特征系数  %.4f \n",  regr.Coef    )
}


// 绘制图像，包括原始数据、真实直线、训练后模型拟合的直线（这里简单用最终参数绘制近似直线示意）
func plotData(xData, yData []float64, labels []int, 
    trueSlope, trueIntercept, a, b, c float64) {
    p := plot.New()
 
    p.Title.Text = "Data Distribution and Fitted Line"
    p.X.Label.Text = "X"
    p.Y.Label.Text = "Y"

    // 绘制真实直线
    // trueLineData := make(plotter.XYs, 2)
    // trueLineData[0] = plotter.XY{X: 0, Y: trueIntercept}
    // trueLineData[1] = plotter.XY{X: 10, Y: trueSlope*10 + trueIntercept}
    // trueLine, err := plotter.NewLine(trueLineData)
    // if err != nil {
    //     panic(err)
    // }
    // trueLine.Color = color.RGBA{R: 0, G: 0, B: 255, A: 255}
    // trueLine.Width = vg.Points(2)
    // p.Add(trueLine)
    // p.Legend.Add("True Line (y=1.342x+2.45)", trueLine)

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
        fitLine.Color = color.RGBA{R: 12, G: 25, B: 0, A: 215}
        fitLine.Width = vg.Points(2)
        p.Add(fitLine)
        p.Legend.Add("Fitted Line", fitLine)
    }

    if err := p.Save(10*vg.Inch, 8*vg.Inch, "result_plot111.png"); err != nil {
        panic(err)
    }
    fmt.Println("图像已保存为 result_plot111.png")
}