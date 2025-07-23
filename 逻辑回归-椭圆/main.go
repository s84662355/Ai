package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
	"github.com/hajimehoshi/ebiten/v2/vector"
)

const (
	screenWidth  = 800 // 窗口宽度
	screenHeight = 600 // 窗口高度
	scale        = 80  // 坐标缩放比例（数学坐标→像素）
	offsetX      = 400 // X轴中心偏移
	offsetY      = 300 // Y轴中心偏移
	trainStep    = 100 // 每帧训练步数
)

// Point 带标签的坐标点
type Point struct {
	X     float64
	Y     float64
	Label int // 1: 曲线内, 0: 曲线外
}

// Params 椭圆方程参数 Ax² + By² + Dx + Ey + F = 0
type Params struct {
	A float64
	B float64
	D float64
	E float64
	F float64
}

// Game 游戏状态
type Game struct {
	points []Point // 所有点数据
	params Params  // 当前拟合参数
	epoch  int     // 当前迭代次数
	loss   float64 // 当前损失值
}

func main() {
	// 初始化点数据
	points := []Point{
		{0.50, 1.81, 1},
		{-1.13, -1.16, 1},
		{1.79, -2.42, 0},
		{1.22, -0.33, 1},
		{-0.88, 2.27, 0},
		{2.19, -2.18, 0},
		{2.65, -0.79, 0},
		{2.43, -2.70, 0},
		{1.92, 0.29, 1},
		{2.13, -1.56, 0},
		{-1.96, 2.02, 0},
		{-0.38, 0.69, 1},
		{0.69, -2.49, 0},
		{1.70, -0.09, 1},
		{1.12, 1.28, 1},
		{-0.36, -1.01, 1},
		{0.22, -1.60, 1},
		{-1.61, -0.80, 1},
		{-1.61, -2.66, 0},
		{0.97, 1.65, 1},
		{2.06, 1.69, 0},
		{2.72, 0.74, 0},
		{1.54, 1.28, 0},
		{0.65, 2.14, 0},
		{-0.31, -0.24, 1},
		{1.20, -1.22, 1},
		{-2.28, 1.66, 0},
		{2.87, 2.34, 0},
		{1.78, 2.44, 0},
		{2.79, -2.43, 0},
		{-0.23, -1.07, 1},
		{-1.13, -1.16, 1},
		{-0.66, -2.49, 0},
		{-1.40, 0.14, 1},
		{2.31, -1.89, 0},
		{-1.53, 1.56, 0},
		{2.51, 1.31, 0},
		{1.37, 1.29, 1},
		{1.87, 0.68, 1},
		{2.58, -2.56, 0},
		{1.85, -1.87, 0},
		{1.47, -0.75, 1},
		{1.43, -2.80, 0},
		{0.09, -2.77, 0},
		{-2.90, -1.67, 0},
		{0.74, 1.89, 0},
		{2.08, 1.09, 0},
		{1.18, -0.82, 1},
		{-1.34, 2.80, 0},
		{-0.07, -0.91, 1},
		{0.94, -0.45, 1},
		{1.69, -1.11, 0},
		{-1.30, 0.94, 1},
		{2.88, -1.87, 0},
		{-0.75, 1.63, 1},
		{2.17, 1.36, 0},
		{1.28, -0.77, 1},
		{2.32, -2.82, 0},
		{-0.10, 1.73, 1},
		{-0.55, 1.48, 1},
		{0.28, 0.47, 1},
		{1.65, 1.41, 0},
		{0.64, 0.19, 1},
		{2.29, 1.04, 0},
		{-2.59, 2.65, 0},
		{0.92, -1.78, 0},
		{1.31, 0.16, 1},
		{-0.74, 1.03, 1},
		{2.10, -0.23, 0},
		{2.36, -0.05, 0},
		{-1.64, -2.53, 0},
		{-1.80, 1.48, 0},
		{1.88, 0.51, 1},
		{-0.95, 0.15, 1},
		{-0.75, -2.69, 0},
		{-0.84, -2.99, 0},
		{1.09, 2.26, 0},
		{-2.75, 2.13, 0},
		{1.91, -2.33, 0},
		{-0.93, 0.20, 1},
		{-2.80, -1.33, 0},
		{0.13, -0.07, 1},
		{0.52, 0.12, 1},
		{-0.55, -1.54, 1},
		{-2.91, 0.93, 0},
		{2.98, 0.40, 0},
		{0.72, 2.67, 0},
		{-2.95, -0.67, 0},
		{2.88, 1.64, 0},
		{1.51, -2.59, 0},
		{-2.92, -1.05, 0},
		{2.81, 1.70, 0},
		{-1.79, 2.67, 0},
		{1.22, 0.17, 1},
		{-0.82, 0.58, 1},
		{-2.87, 2.89, 0},
		{1.43, -0.33, 1},
		{0.01, 0.93, 1},
		{2.01, 0.50, 0},
		{-1.83, -1.62, 0},
	}

	// 初始化参数
	rand.Seed(time.Now().UnixNano())
	initParams := Params{
		A: rand.Float64()*0.2 - 0.1,
		B: rand.Float64()*0.2 - 0.1,
		D: rand.Float64()*0.2 - 0.1,
		E: rand.Float64()*0.2 - 0.1,
		F: rand.Float64()*0.2 - 0.1,
	}

	// 初始化游戏
	game := &Game{
		points: points,
		params: initParams,
		epoch:  0,
		loss:   computeLoss(points, initParams),
	}

	// 配置窗口
	ebiten.SetWindowSize(screenWidth, screenHeight)
	ebiten.SetWindowTitle("椭圆拟合动画")
	if err := ebiten.RunGame(game); err != nil {
		fmt.Printf("启动失败: %v\n", err)
	}
}

// Update 更新游戏状态（训练模型）
func (g *Game) Update() error {
	// 每帧执行多次训练步骤
	for i := 0; i < trainStep; i++ {
		dA, dB, dD, dE, dF := computeGradient(g.points, g.params)
		lr := 0.0001 // 学习率
		g.params.A -= lr * dA
		g.params.B -= lr * dB
		g.params.D -= lr * dD
		g.params.E -= lr * dE
		g.params.F -= lr * dF
		g.epoch++
	}
	// 更新损失值
	g.loss = computeLoss(g.points, g.params)
	return nil
}

// Draw 绘制画面
func (g *Game) Draw(screen *ebiten.Image) {
	// 填充背景
	screen.Fill(color.RGBA{30, 30, 30, 255})

	// 绘制坐标轴
	drawAxes(screen)

	// 绘制所有点
	drawPoints(screen, g.points)

	// 绘制当前拟合曲线
	drawCurve(screen, g.params)

	// 显示训练信息
	drawInfo(screen, g.epoch, g.loss, g.params)

	// 绘制圆边框
	vector.StrokeCircle(
		screen,
		offsetX,
		offsetY,
		2*scale,
		3,                            // 线宽
		color.RGBA{23, 80, 200, 255}, // 边框色（深蓝色）
		false,
	)
}

// Layout 设置窗口布局
func (g *Game) Layout(_, _ int) (int, int) {
	return screenWidth, screenHeight
}

// 绘制坐标轴
func drawAxes(screen *ebiten.Image) {
	// X轴
	vector.StrokeLine(screen, 0, float32(offsetY), float32(screenWidth), float32(offsetY), 1, color.RGBA{100, 100, 100, 255}, false)
	// Y轴
	vector.StrokeLine(screen, float32(offsetX), 0, float32(offsetX), float32(screenHeight), 1, color.RGBA{100, 100, 100, 255}, false)

	// 绘制刻度
	for x := -5; x <= 5; x++ {
		px := float32(x)*scale + float32(offsetX)
		vector.StrokeLine(screen, px, float32(offsetY)-3, px, float32(offsetY)+3, 1, color.RGBA{150, 150, 150, 255}, false)
	}
	for y := -5; y <= 5; y++ {
		py := float32(-y)*scale + float32(offsetY)
		vector.StrokeLine(screen, float32(offsetX)-3, py, float32(offsetX)+3, py, 1, color.RGBA{150, 150, 150, 255}, false)
	}
}

// 绘制所有点
func drawPoints(screen *ebiten.Image, points []Point) {
	for _, pt := range points {
		// 坐标转换：数学坐标 -> 屏幕坐标
		x := float64(pt.X*scale) + float64(offsetX)
		y := float64(-pt.Y*scale) + float64(offsetY) // Y轴翻转

		// 设置颜色（内点绿色，外点红色）
		c := color.RGBA{255, 0, 0, 255}
		if pt.Label == 1 {
			c = color.RGBA{0, 255, 0, 255}
		}

		// 绘制点（小矩形）
		ebitenutil.DrawRect(screen, x-2, y-2, 4, 4, c)
	}
}

// 绘制拟合曲线
func drawCurve(screen *ebiten.Image, p Params) {
	// 生成曲线上的点（网格采样）
	const step = 0.02
	var points []struct{ x, y float32 }

	// 扩大搜索范围确保捕获完整曲线
	for x := -4.0; x <= 4.0; x += step {
		for y := -4.0; y <= 4.0; y += step {
			val := p.A*x*x + p.B*y*y + p.D*x + p.E*y + p.F
			if math.Abs(val) < 0.05 { // 接近曲线的点
				sx := float32(x*scale) + float32(offsetX)
				sy := float32(-y*scale) + float32(offsetY)
				points = append(points, struct{ x, y float32 }{sx, sy})
			}
		}
	}

	// 绘制曲线（连接相邻点）
	if len(points) > 1 {
		for i := 1; i < len(points); i++ {
			// 只连接距离近的点（避免乱线）
			dx := points[i].x - points[i-1].x
			dy := points[i].y - points[i-1].y
			if dx*dx+dy*dy < 20*20 {
				ebitenutil.DrawRect(screen, float64(points[i-1].x), float64(points[i-1].y), 2, 2, color.RGBA{112, 105, 145, 255})

				// vector.StrokeLine(
				// 	screen,
				// 	points[i-1].x, points[i-1].y,
				// 	points[i].x, points[i].y,
				// 	2, color.RGBA{0, 255, 255, 255}, true,
				// )
			}
		}
	}
}

// 绘制训练信息
func drawInfo(screen *ebiten.Image, epoch int, loss float64, p Params) {
	info := fmt.Sprintf(
		"epoch: %d\n loss: %.6f\n model: %.4fx² + %.4fy² + %.4fx + %.4fy + %.4f = 0",
		epoch, loss, p.A, p.B, p.D, p.E, p.F,
	)
	ebitenutil.DebugPrint(screen, info)
	fmt.Printf(
		"epoch: %d  loss: %.6f  model: %.4fx² + %.4fy² + %.4fx + %.4fy + %.4f = 0 \n",
		epoch, loss, p.A, p.B, p.D, p.E, p.F,
	)
}

// sigmoid 函数映射概率
func sigmoid(z float64) float64 {
	return 1 / (1 + math.Exp(-z))
}

// 计算交叉熵损失
func computeLoss(points []Point, p Params) float64 {
	var loss float64
	for _, pt := range points {
		val := p.A*pt.X*pt.X + p.B*pt.Y*pt.Y + p.D*pt.X + p.E*pt.Y + p.F
		z := -val
		prob := sigmoid(z)

		// 防止log(0)错误
		if prob < 1e-10 {
			prob = 1e-10
		}
		if prob > 1-1e-10 {
			prob = 1 - 1e-10
		}

		loss -= float64(pt.Label)*math.Log(prob) + (1-float64(pt.Label))*math.Log(1-prob)
	}
	return loss / float64(len(points))
}

// 计算梯度向量
func computeGradient(points []Point, p Params) (dA, dB, dD, dE, dF float64) {
	var gradA, gradB, gradD, gradE, gradF float64
	for _, pt := range points {
		val := p.A*pt.X*pt.X + p.B*pt.Y*pt.Y + p.D*pt.X + p.E*pt.Y + p.F
		z := -val
		prob := sigmoid(z)

		// 各参数偏导数
		gradA += (prob - float64(pt.Label)) * (-pt.X * pt.X)
		gradB += (prob - float64(pt.Label)) * (-pt.Y * pt.Y)
		gradD += (prob - float64(pt.Label)) * (-pt.X)
		gradE += (prob - float64(pt.Label)) * (-pt.Y)
		gradF += (prob - float64(pt.Label)) * (-1)
	}
	// 平均梯度
	n := float64(len(points))
	return gradA / n, gradB / n, gradD / n, gradE / n, gradF / n
}

// 梯度下降训练
func train(points []Point, init Params, lr float64, epochs int) Params {
	params := init

	for i := 0; i < epochs; i++ {
		dA, dB, dD, dE, dF := computeGradient(points, params)
		params.A -= lr * dA
		params.B -= lr * dB
		params.D -= lr * dD
		params.E -= lr * dE
		params.F -= lr * dF

		// 每2000轮输出损失
		if i%2000 == 0 {
			loss := computeLoss(points, params)
			fmt.Printf("Epoch %d, Loss: %.6f\n", i, loss)
		}
	}

	return params
}
