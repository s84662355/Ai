package main

import (
	"image"
	"image/color"
	"image/draw"
	"image/gif"
 
	"math"
	"math/rand"
	"os"
	"time"

	"fmt"

	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

// 逻辑斯蒂函数
func logistic(a, b, c, x, y float64) float64 {
	g := a + b*x + c*y
	return 1.0 / (1.0 + math.Exp(-g))
}

// 交叉熵损失计算
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

// 梯度计算函数
func gradientA(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
	m := len(yTrue)
	sum := 0.0
	for i := range yTrue {
		p := logistic(a, b, c, xData[i], yData[i])
		sum += (p - float64(yTrue[i]))
	}
	return sum / float64(m)
}

func gradientB(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
	m := len(yTrue)
	sum := 0.0
	for i := range yTrue {
		p := logistic(a, b, c, xData[i], yData[i])
		sum += (p - float64(yTrue[i])) * xData[i]
	}
	return sum / float64(m)
}

func gradientC(yTrue []int, xData, yData []float64, a, b, c float64) float64 {
	m := len(yTrue)
	sum := 0.0
	for i := range yTrue {
		p := logistic(a, b, c, xData[i], yData[i])
		sum += (p - float64(yTrue[i])) * yData[i]
	}
	return sum / float64(m)
}

// 数据生成函数
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

// 绘制单帧图像（通过临时文件规避接口问题）
func plotFrame(xData, yData []float64, labels []int,
	trueSlope, trueIntercept, a, b, c float64, iteration int) image.Image {
	// 创建绘图对象
	p := plot.New()
	p.Title.Text = fmt.Sprintf("Fitting Process (Iteration: %d)", iteration)
	p.X.Label.Text = "X"
	p.Y.Label.Text = "Y"
	p.X.Min = -1
	p.X.Max = 11
	p.Y.Min = -5
	p.Y.Max = 20

	// 绘制真实直线（蓝色）
	trueLineData := plotter.XYs{
		{X: 0, Y: trueIntercept},
		{X: 10, Y: trueSlope*10 + trueIntercept},
	}
	trueLine, _ := plotter.NewLine(trueLineData)
	trueLine.Color = color.RGBA{B: 255, A: 255}
	trueLine.Width = vg.Points(2)
	p.Add(trueLine)
	p.Legend.Add("True Line", trueLine)

	// 绘制数据点
	scatterUp := make(plotter.XYs, 0)
	scatterDown := make(plotter.XYs, 0)
	for i := range xData {
		if labels[i] == 1 {
			scatterUp = append(scatterUp, plotter.XY{X: xData[i], Y: yData[i]})
		} else {
			scatterDown = append(scatterDown, plotter.XY{X: xData[i], Y: yData[i]})
		}
	}
	upPlotter, _ := plotter.NewScatter(scatterUp)
	upPlotter.Color = color.RGBA{R: 255, A: 255}
	upPlotter.Radius = vg.Points(2)
	p.Add(upPlotter)
	p.Legend.Add("Class 1", upPlotter)

	downPlotter, _ := plotter.NewScatter(scatterDown)
	downPlotter.Color = color.RGBA{B: 255, A: 255}
	downPlotter.Radius = vg.Points(2)
	p.Add(downPlotter)
	p.Legend.Add("Class 0", downPlotter)

	// 绘制拟合直线（绿色）
	if c != 0 {
		fitLineData := plotter.XYs{
			{X: 0, Y: (-a - b*0) / c},
			{X: 10, Y: (-a - b*10) / c},
		}
		fitLine, _ := plotter.NewLine(fitLineData)
		fitLine.Color = color.RGBA{G: 255, A: 255}
		fitLine.Width = vg.Points(2)
		p.Add(fitLine)
		p.Legend.Add("Fitted Line", fitLine)
	}

	// 关键修正：通过临时文件保存再读取，规避接口不兼容
	tempFile, err := os.CreateTemp("", "frame-*.png")
	if err != nil {
		panic(err)
	}
	tempFileName := tempFile.Name()
	tempFile.Close()
	defer os.Remove(tempFileName) // 清理临时文件

	// 保存图像到临时文件
	if err := p.Save(8*vg.Inch, 6*vg.Inch, tempFileName); err != nil {
		panic(err)
	}

	// 从临时文件读取图像
	f, err := os.Open(tempFileName)
	if err != nil {
		panic(err)
	}
	defer f.Close()
	img, _, err := image.Decode(f)
	if err != nil {
		panic(err)
	}
	return img
}

// 合成GIF动画
func saveGIF(frames []*image.Paletted, delay int, filename string) error {
	f, err := os.Create(filename)
	if err != nil {
		return err
	}
	defer f.Close()

	gifData := &gif.GIF{
		Image: frames,
		Delay: make([]int, len(frames)),
	}
	for i := range gifData.Delay {
		gifData.Delay[i] = delay
	}
	return gif.EncodeAll(f, gifData)
}

// 转换为GIF所需格式
func toPaletted(img image.Image) *image.Paletted {
	bounds := img.Bounds()
	palette := []color.Color{
		color.Transparent,
		color.RGBA{R: 255, G: 0, B: 0, A: 255}, // 红色
		color.RGBA{R: 0, G: 0, B: 255, A: 255}, // 蓝色
		color.RGBA{R: 0, G: 255, B: 0, A: 255}, // 绿色
		color.RGBA{R: 0, G: 0, B: 0, A: 255},   // 黑色（文本）
		color.RGBA{R: 255, G: 255, B: 255, A: 255}, // 白色（背景）
	}
	paletted := image.NewPaletted(bounds, palette)
	draw.Draw(paletted, bounds, img, bounds.Min, draw.Src)
	return paletted
}

func main() {
	// 数据参数
	const (
		n         = 2000
		slope     = 1.342
		intercept = 2.45
		noiseMax  = 3.554646
	)

	xData, yData, yTrue := generateData(n, slope, intercept, noiseMax)

	// 模型参数
	a, b, c := 0.0, 0.0, 0.0
	learningRate := 0.0008
	iterations := 180000
	frameInterval := 100 
	frames := make([]*image.Paletted, 0)

	// 训练并生成帧
	for i := 0; i < iterations; i++ {
		gradA := gradientA(yTrue, xData, yData, a, b, c)
		gradB := gradientB(yTrue, xData, yData, a, b, c)
		gradC := gradientC(yTrue, xData, yData, a, b, c)
		a -= learningRate * gradA
		b -= learningRate * gradB
		c -= learningRate * gradC

		if i%frameInterval == 0 {
			loss := crossEntropyLoss(yTrue, xData, yData, a, b, c)
			fmt.Printf("迭代 %d 次，损失: J=%.4f\n", i, loss)
			img := plotFrame(xData, yData, yTrue, slope, intercept, a, b, c, i)
			frames = append(frames, toPaletted(img))
		}
	}

	fmt.Printf("训练后参数：a=%.4f, b=%.4f, c=%.4f\n", a, b, c)

	if err := saveGIF(frames, 15, "fitting_animation.gif"); err != nil {
		panic(err)
	}
	fmt.Println("动画已保存为 fitting_animation.gif")
}