package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"os"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/text"
	"golang.org/x/image/font"
	"golang.org/x/image/font/opentype"
	"gonum.org/v1/gonum/stat/distuv"
)

const (
	screenWidth  = 1500
	screenHeight = 800
	axisWidth    = 2.0
	xAxisYPos    = screenHeight / 2.0
	yAxisXPos    = screenWidth / 2.0
	// 数学坐标系范围
	xMin = -10.0
	xMax = 10.0
	yMin = -20.0
	yMax = 100.0
	// 控制更新间隔（秒）
	updateInterval = 0.001
)

// 颜色定义
var (
	axisColor     = color.RGBA{255, 255, 255, 255}
	pointColor    = color.RGBA{255, 255, 255, 128}
	trueLineColor = color.RGBA{0, 255, 0, 255}
	fitLineColor  = color.RGBA{165, 120, 32, 255}
	gridColor     = color.RGBA{100, 100, 100, 60}
	labelColor    = color.RGBA{255, 255, 255, 255}
	progressColor = color.RGBA{0, 150, 255, 255}
)

// 全局状态
var (
	dataSize      int = 2000
	data          [][]float64
	a, b, c       float64
	lr            = 0.0001
	numIterations = 50000
	step          = 0
	ttfFont       font.Face
	lastUpdate    time.Time                      // 记录上次更新时间
	ta, tb, tc    float64   = 1.234, -3.54, 2.45 // 真实参数
	Sigma         float64   = 2.0                // 噪声水平
)

// 初始化随机数生成器
func init() {
	rand.Seed(time.Now().UnixNano())
}

// 生成符合二次函数的样本数据
func initData(numSamples int) {
	data = make([][]float64, 0, numSamples)
	for i := 0; i < numSamples; i++ {
		x := rand.Float64()*(xMax-xMin) + xMin
		eps := distuv.Normal{Mu: 0, Sigma: Sigma}.Rand() // 高斯噪声
		y := ta*x*x + tb*x + tc + eps                    // 真实函数+噪声
		data = append(data, []float64{x, y})
	}
}

// 计算二次函数的均方误差
func Mse(a, b, c float64, points [][]float64) float64 {
	totalError := 0.0
	for _, p := range points {
		x, y := p[0], p[1]
		// 计算预测值与实际值的误差平方
		totalError += math.Pow(y-(a*x*x+b*x+c), 2)
	}
	return totalError / float64(len(points))
}

// 执行一次梯度下降步骤，更新参数
func StepGradient(a, b, c float64, points [][]float64, lr float64) (float64, float64, float64) {
	aGrad, bGrad, cGrad := 0.0, 0.0, 0.0
	M := float64(len(points))

	for _, p := range points {
		x, y := p[0], p[1]
		err := a*x*x + b*x + c - y

		// 计算各参数的梯度
		aGrad += (2 / M) * x * x * err
		bGrad += (2 / M) * x * err
		cGrad += (2 / M) * err
	}

	// 更新参数
	return a - lr*aGrad, b - lr*bGrad, c - lr*cGrad
}

// 游戏结构
type Game struct{}

// Update 控制更新节奏，定期执行梯度下降
func (g *Game) Update() error {
	// 首次运行初始化时间
	if lastUpdate.IsZero() {
		lastUpdate = time.Now()
		return nil
	}

	// 检查是否达到更新间隔
	now := time.Now()
	if now.Sub(lastUpdate) >= time.Duration(updateInterval*float64(time.Second)) && step < numIterations {
		a, b, c = StepGradient(a, b, c, data, lr)
		step++
		if step%100 == 0 {
			loss := Mse(a, b, c, data)
			fmt.Printf("Iteration:%d, loss:%f, a:%f, b:%f, c:%f\n", step, loss, a, b, c)
		}
		lastUpdate = now
	}
	return nil
}

// Draw 绘制当前状态
func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{0, 0, 0, 255})

	// 绘制网格和坐标轴
	drawGrid(screen)
	drawAxis(screen)
	drawAxisLabels(screen)

	// 绘制样本点
	drawPoints(screen, data)

	// 绘制当前拟合曲线
	drawCurve(screen, a, b, c, fitLineColor)

	// 绘制真实曲线
	drawCurve(screen, ta, tb, tc, trueLineColor)

	// 绘制图例
	drawLegend(screen)

	// 绘制当前参数和损失
	drawStats(screen)

	// 绘制进度条
	drawProgressBar(screen)
}

// Layout 返回屏幕尺寸
func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}

// 绘制网格线
func drawGrid(screen *ebiten.Image) {
	// 绘制垂直网格线
	for x := xMin; x <= xMax; x += 1.0 {
		xScreen := math.Round((x - xMin) / (xMax - xMin) * screenWidth)
		ebitenutil.DrawLine(screen, xScreen, 0, xScreen, screenHeight, gridColor)
	}

	// 绘制水平网格线
	for y := yMin; y <= yMax; y += 5.0 {
		yScreen := math.Round((yMax - y) / (yMax - yMin) * screenHeight)
		ebitenutil.DrawLine(screen, 0, yScreen, screenWidth, yScreen, gridColor)
	}
}

// 绘制坐标轴
func drawAxis(screen *ebiten.Image) {
	// 绘制 x 轴
	ebitenutil.DrawLine(screen, 0, xAxisYPos, screenWidth, xAxisYPos, axisColor)
	// 绘制 y 轴
	ebitenutil.DrawLine(screen, yAxisXPos, 0, yAxisXPos, screenHeight, axisColor)

	// 绘制箭头
	arrowSize := 10.0
	// x 轴箭头
	ebitenutil.DrawLine(screen, screenWidth-arrowSize, xAxisYPos-arrowSize/2, screenWidth, xAxisYPos, axisColor)
	ebitenutil.DrawLine(screen, screenWidth-arrowSize, xAxisYPos+arrowSize/2, screenWidth, xAxisYPos, axisColor)
	// y 轴箭头
	ebitenutil.DrawLine(screen, yAxisXPos-arrowSize/2, arrowSize, yAxisXPos, 0, axisColor)
	ebitenutil.DrawLine(screen, yAxisXPos+arrowSize/2, arrowSize, yAxisXPos, 0, axisColor)
}

// 绘制坐标轴标签
func drawAxisLabels(screen *ebiten.Image) {
	// 绘制 x 轴刻度
	for x := xMin; x <= xMax; x += 2.0 {
		xScreen := math.Round((x - xMin) / (xMax - xMin) * screenWidth)
		ebitenutil.DrawLine(screen, xScreen, xAxisYPos-5, xScreen, xAxisYPos+5, axisColor)
		if x != 0 { // 0 点已被 y 轴标签覆盖
			label := fmt.Sprintf("%.0f", x)
			if ttfFont != nil {
				text.Draw(screen, label, ttfFont, int(xScreen)-5, int(xAxisYPos)+20, labelColor)
			} else {
				ebitenutil.DebugPrintAt(screen, label, int(xScreen)-5, int(xAxisYPos)+15)
			}
		}
	}

	// 绘制 y 轴刻度
	for y := yMin; y <= yMax; y += 10.0 {
		yScreen := math.Round((yMax - y) / (yMax - yMin) * screenHeight)
		ebitenutil.DrawLine(screen, yAxisXPos-5, yScreen, yAxisXPos+5, yScreen, axisColor)
		if y != 0 { // 0 点已被 x 轴标签覆盖
			label := fmt.Sprintf("%.0f", y)
			if ttfFont != nil {
				text.Draw(screen, label, ttfFont, int(yAxisXPos)+10, int(yScreen)+5, labelColor)
			} else {
				ebitenutil.DebugPrintAt(screen, label, int(yAxisXPos)+10, int(yScreen)-5)
			}
		}
	}

	// 绘制轴标签
	if ttfFont != nil {
		text.Draw(screen, "x", ttfFont, screenWidth-15, int(xAxisYPos)-15, labelColor)
		text.Draw(screen, "y", ttfFont, int(yAxisXPos)+15, 20, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, "x", screenWidth-15, int(xAxisYPos)-15)
		ebitenutil.DebugPrintAt(screen, "y", int(yAxisXPos)+15, 20)
	}
}

// 绘制数据点
func drawPoints(screen *ebiten.Image, points [][]float64) {
	for _, p := range points {
		x, y := p[0], p[1]
		xScreen := (x - xMin) / (xMax - xMin) * screenWidth
		yScreen := (yMax - y) / (yMax - yMin) * screenHeight
		ebitenutil.DrawCircle(screen, xScreen, yScreen, 2, pointColor)
	}
}

// 绘制二次曲线
func drawCurve(screen *ebiten.Image, a, b, c float64, color color.Color) {
	// 绘制足够密集的点来近似曲线
	step := (xMax - xMin) / 1000
	for x := xMin; x < xMax; x += step {
		y := a*x*x + b*x + c
		x1Screen := (x - xMin) / (xMax - xMin) * screenWidth
		y1Screen := (yMax - y) / (yMax - yMin) * screenHeight

		x2 := x + step
		y2 := a*x2*x2 + b*x2 + c
		x2Screen := (x2 - xMin) / (xMax - xMin) * screenWidth
		y2Screen := (yMax - y2) / (yMax - yMin) * screenHeight

		ebitenutil.DrawLine(screen, x1Screen, y1Screen, x2Screen, y2Screen, color)
	}
}

// 绘制图例
func drawLegend(screen *ebiten.Image) {
	legendX := 20
	legendY := 20
	legendSpacing := 20

	// 绘制图例标题
	if ttfFont != nil {
		text.Draw(screen, "Legend:", ttfFont, legendX, legendY, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, "Legend:", legendX, legendY)
	}

	// 绘制真实曲线图例
	ebitenutil.DrawLine(screen, float64(legendX), float64(legendY+legendSpacing),
		float64(legendX+30), float64(legendY+legendSpacing), trueLineColor)
	if ttfFont != nil {
		text.Draw(screen, fmt.Sprintf("True Curve (y=%.3fx²+%.3fx+%.3f)", ta, tb, tc), ttfFont, legendX+40, legendY+legendSpacing+5, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("True Curve (y=%.3fx²+%.3fx+%.3f)", ta, tb, tc), legendX+40, legendY+legendSpacing-5)
	}

	// 绘制拟合曲线图例
	ebitenutil.DrawLine(screen, float64(legendX), float64(legendY+legendSpacing*2),
		float64(legendX+30), float64(legendY+legendSpacing*2), fitLineColor)
	if ttfFont != nil {
		text.Draw(screen, fmt.Sprintf("Fit Curve (y=%.3fx²+%.3fx+%.3f)", a, b, c), ttfFont, legendX+40, legendY+legendSpacing*2+5, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Fit Curve (y=%.3fx²+%.3fx+%.3f)", a, b, c), legendX+40, legendY+legendSpacing*2-5)
	}
}

// 绘制统计信息
func drawStats(screen *ebiten.Image) {
	statsX := 20
	statsY := screenHeight - 140
	statsSpacing := 20

	loss := Mse(a, b, c, data)
	progress := float64(step) / float64(numIterations) * 100

	if ttfFont != nil {
		text.Draw(screen, "Training Progress:", ttfFont, statsX, statsY, labelColor)
		text.Draw(screen, fmt.Sprintf("Iteration: %d/%d (%.1f%%)", step, numIterations, progress), ttfFont, statsX, statsY+statsSpacing, labelColor)
		text.Draw(screen, fmt.Sprintf("Loss: %.6f", loss), ttfFont, statsX, statsY+statsSpacing*2, labelColor)
		text.Draw(screen, fmt.Sprintf("Parameters: a=%.6f, b=%.6f, c=%.6f", a, b, c), ttfFont, statsX, statsY+statsSpacing*3, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, "Training Progress:", statsX, statsY)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Iteration: %d/%d (%.1f%%)", step, numIterations, progress), statsX, statsY+statsSpacing)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Loss: %.6f", loss), statsX, statsY+statsSpacing*2)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Parameters: a=%.6f, b=%.6f, c=%.6f", a, b, c), statsX, statsY+statsSpacing*3)
	}
}

// 绘制进度条
func drawProgressBar(screen *ebiten.Image) {
	barX := 20
	barY := screenHeight - 30
	barWidth := screenWidth - 40
	barHeight := 15

	// 绘制进度条背景
	ebitenutil.DrawRect(screen, float64(barX), float64(barY), float64(barWidth), float64(barHeight), color.RGBA{50, 50, 50, 255})

	// 计算进度条长度
	progress := float64(step) / float64(numIterations)
	progressWidth := float64(barWidth) * progress

	// 绘制进度条
	ebitenutil.DrawRect(screen, float64(barX), float64(barY), progressWidth, float64(barHeight), progressColor)

	// 绘制进度文本
	progressText := fmt.Sprintf("Training: %.1f%%", progress*100)
	if ttfFont != nil {
		text.Draw(screen, progressText, ttfFont, barX+10, barY+12, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, progressText, barX+10, barY+5)
	}
}

// 加载字体
func loadFont() {
	// 尝试加载系统字体
	// 注意：实际使用时可能需要提供具体的字体文件路径
	fontBytes, err := os.ReadFile("arial.ttf")
	if err != nil {
		fmt.Println("Failed to load font, using default:", err)
		return
	}

	f, err := opentype.Parse(fontBytes)
	if err != nil {
		fmt.Println("Failed to parse font:", err)
		return
	}

	ttfFont, err = opentype.NewFace(f, &opentype.FaceOptions{
		Size:    12,
		DPI:     72,
		Hinting: font.HintingFull,
	})
	if err != nil {
		fmt.Println("Failed to create font face:", err)
	}
}

func main() {
	// 设置最大帧率
	ebiten.SetMaxTPS(30)

	// 尝试加载字体（可选）
	loadFont()

	// 初始化数据、参数
	initData(dataSize)
	a, b, c = 0.0, 0.0, 0.0
	lastUpdate = time.Now()

	// 启动可视化
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("二次函数拟合可视化")
	if err := ebiten.RunGame(&Game{}); err != nil {
		panic(err)
	}
}
