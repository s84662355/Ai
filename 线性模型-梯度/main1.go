package main

import (
	"fmt"
	"image/color"
	"math"
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
	xMin = -30.0
	xMax = 30.0
	yMin = -20.0
	yMax = 20.0
	// 控制更新间隔（秒）
	updateInterval = 0.0001
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
	w, b          float64
	lr            = 0.0005
	numIterations = 200000
	step          = 0
	ttfFont       font.Face
	lastUpdate    time.Time // 记录上次更新时间
	tw, tb        float64   = 1.72212862, 2.65145218
	Sigma         float64   = 1.548564
)

func initData(numSamples int) {
	data = make([][]float64, 0, numSamples)
	for i := 0; i < numSamples; i++ {
		x := distuv.Uniform{Min: xMin, Max: xMax}.Rand()
		eps := distuv.Normal{Mu: 0, Sigma: Sigma}.Rand()
		y := tw*x + tb + eps
		data = append(data, []float64{x, y})
	}
}

func Mse(b, w float64, points [][]float64) float64 {
	totalError := 0.0
	for _, p := range points {
		x, y := p[0], p[1]
		totalError += math.Pow(y-(w*x+b), 2)
	}
	return totalError / float64(len(points))
}

func StepGradient(b, w float64, points [][]float64, lr float64) (float64, float64) {
	bGrad, wGrad := 0.0, 0.0
	M := float64(len(points))
	for _, p := range points {
		x, y := p[0], p[1]
		err := w*x + b - y
		bGrad += (2 / M) * err
		wGrad += (2 / M) * x * err
	}
	return b - lr*bGrad, w - lr*wGrad
}

type Game struct{}

// Update 控制更新节奏，每0.5秒执行一次梯度下降
func (g *Game) Update() error {
	// 首次运行初始化时间
	if lastUpdate.IsZero() {
		lastUpdate = time.Now()
		return nil
	}

	// 检查是否达到更新间隔
	now := time.Now()
	if now.Sub(lastUpdate) >= time.Duration(updateInterval*float64(time.Second)) && step < numIterations {
		b, w = StepGradient(b, w, data, lr)
		step++
		if step%2 == 0 {
			loss := Mse(b, w, data)
			fmt.Printf("Iteration:%d, loss:%f, w:%f, b:%f\n", step, loss, w, b)
		}
		lastUpdate = now
	}
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	screen.Fill(color.RGBA{0, 0, 0, 255})

	// 绘制网格和坐标轴
	drawGrid(screen)
	drawAxis(screen)
	drawAxisLabels(screen)

	// 绘制样本点
	drawPoints(screen, data)

	// 绘制当前拟合直线
	drawLine(screen, w, b, fitLineColor)

	// 绘制真实直线
	drawLine(screen, tw, tb, trueLineColor)

	// 绘制图例
	drawLegend(screen)

	// 绘制当前参数和损失
	drawStats(screen)

	// 绘制进度条
	drawProgressBar(screen)
}

func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}

func drawGrid(screen *ebiten.Image) {
	// 绘制垂直网格线
	for x := xMin; x <= xMax; x += 1.0 {
		xScreen := math.Round((x - xMin) / (xMax - xMin) * screenWidth)
		ebitenutil.DrawLine(screen, xScreen, 0, xScreen, screenHeight, gridColor)
	}

	// 绘制水平网格线
	for y := yMin; y <= yMax; y += 1.0 {
		yScreen := math.Round((yMax - y) / (yMax - yMin) * screenHeight)
		ebitenutil.DrawLine(screen, 0, yScreen, screenWidth, yScreen, gridColor)
	}
}

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
	for y := yMin; y <= yMax; y += 2.0 {
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

func drawPoints(screen *ebiten.Image, points [][]float64) {
	for _, p := range points {
		x, y := p[0], p[1]
		xScreen := (x - xMin) / (xMax - xMin) * screenWidth
		yScreen := (yMax - y) / (yMax - yMin) * screenHeight
		ebitenutil.DrawCircle(screen, xScreen, yScreen, 2, pointColor)
	}
}

func drawLine(screen *ebiten.Image, w, b float64, c color.Color) {
	x1, y1 := xMin, w*xMin+b
	x2, y2 := xMax, w*xMax+b

	x1Screen := (x1 - xMin) / (xMax - xMin) * screenWidth
	y1Screen := (yMax - y1) / (yMax - yMin) * screenHeight
	x2Screen := (x2 - xMin) / (xMax - xMin) * screenWidth
	y2Screen := (yMax - y2) / (yMax - yMin) * screenHeight

	ebitenutil.DrawLine(screen, x1Screen, y1Screen, x2Screen, y2Screen, c)
}

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

	// 绘制真实直线图例
	ebitenutil.DrawLine(screen, float64(legendX), float64(legendY+legendSpacing),
		float64(legendX+30), float64(legendY+legendSpacing), trueLineColor)
	if ttfFont != nil {
		text.Draw(screen, fmt.Sprintf("True Line (y=%.8fx+%.8f)", tw, tb), ttfFont, legendX+40, legendY+legendSpacing+5, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("True Line (y=%.8fx+%.8f)", tw, tb), legendX+40, legendY+legendSpacing-5)
	}

	// 绘制拟合直线图例
	ebitenutil.DrawLine(screen, float64(legendX), float64(legendY+legendSpacing*2),
		float64(legendX+30), float64(legendY+legendSpacing*2), fitLineColor)
	if ttfFont != nil {
		text.Draw(screen, fmt.Sprintf("Fit Line (y=%.8fx+%.8f)", w, b), ttfFont, legendX+40, legendY+legendSpacing*2+5, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Fit Line (y=%.8fx+%.8f)", w, b), legendX+40, legendY+legendSpacing*2-5)
	}
}

func drawStats(screen *ebiten.Image) {
	statsX := 20
	statsY := screenHeight - 140
	statsSpacing := 20

	loss := Mse(b, w, data)
	progress := float64(step) / float64(numIterations) * 100

	if ttfFont != nil {
		text.Draw(screen, "Training Progress:", ttfFont, statsX, statsY, labelColor)
		text.Draw(screen, fmt.Sprintf("Iteration: %d/%d (%.1f%%)", step, numIterations, progress), ttfFont, statsX, statsY+statsSpacing, labelColor)
		text.Draw(screen, fmt.Sprintf("Loss: %.6f", loss), ttfFont, statsX, statsY+statsSpacing*2, labelColor)
		text.Draw(screen, fmt.Sprintf("Parameters: w=%.8f, b=%.8f", w, b), ttfFont, statsX, statsY+statsSpacing*3, labelColor)
	} else {
		ebitenutil.DebugPrintAt(screen, "Training Progress:", statsX, statsY)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Iteration: %d/%d (%.1f%%)", step, numIterations, progress), statsX, statsY+statsSpacing)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Loss: %.6f", loss), statsX, statsY+statsSpacing*2)
		ebitenutil.DebugPrintAt(screen, fmt.Sprintf("Parameters: w=%.8f, b=%.8f", w, b), statsX, statsY+statsSpacing*3)
	}
}

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
	// 设置最大帧率，避免CPU占用过高
	ebiten.SetMaxTPS(30) // 每秒最多30帧，足够流畅显示

	// 尝试加载字体（可选）
	loadFont()

	// 初始化数据、参数
	initData(dataSize)
	w, b = 0.0, 0.0
	lastUpdate = time.Now()

	// 启动 Ebiten 可视化
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("Linear Regression Visualization")
	if err := ebiten.RunGame(&Game{}); err != nil {
		panic(err)
	}
}
