package main

import (
    "fmt"
    "image/color"
    "math"
    "math/rand"
    "time"

    "github.com/hajimehoshi/ebiten/v2"
    "github.com/hajimehoshi/ebiten/v2/ebitenutil"
    "github.com/hajimehoshi/ebiten/v2/text"
    "golang.org/x/image/font/basicfont"
)

const (
    screenWidth  = 1200
    screenHeight = 800
    // 坐标轴范围
    xMinPlot = -30.0
    xMaxPlot = 30.0
    yMinPlot = -20.0
    yMaxPlot = 20.0
    // 数据相关
    numPoints = 2000
    trueSlope = 1.342
    trueIntercept = 2.45
    noiseMax  = 3.554646
    // 训练相关
    learningRate = 0.006
    totalIterations = 200000
)

// 全局状态
var (
    // 数据存储
    xData, yData []float64
    labels       []int

    // 模型参数
    a, b, c float64 = 0.0, 0.0, 0.0
    currentIter   int
    loss          float64

    // 颜色定义
    colorBg          = color.RGBA{R: 0, G: 0, B: 0, A: 255}         // 背景黑色
    colorGrid        = color.RGBA{R: 100, G: 100, B: 100, A: 128}   // 网格灰色半透明
    colorTrueLine    = color.RGBA{R: 0, G: 255, B: 0, A: 255}       // 真实边界绿色
    colorFitLine     = color.RGBA{R: 255, G: 165, B: 0, A: 255}     // 拟合边界橙色
    colorClass0      = color.RGBA{R: 0, G: 0, B: 255, A: 255}       // 类别0蓝色
    colorClass1      = color.RGBA{R: 255, G: 0, B: 0, A: 255}       // 类别1红色
    colorText        = color.RGBA{R: 255, G: 255, B: 255, A: 255}   // 文字白色
)

// 计算逻辑斯蒂函数
func logistic(a, b, c, x, y float64) float64 {
    g := a + b*x + c*y
    return 1.0 / (1.0 + math.Exp(-g))
}

// 计算交叉熵损失
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

// 计算梯度
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

// 生成数据
func generateData() {
    rand.Seed(time.Now().UnixNano())
    xData = make([]float64, numPoints)
    yData = make([]float64, numPoints)
    labels = make([]int, numPoints)

    for i := 0; i < numPoints; i++ {
        xData[i] = rand.Float64()*(xMaxPlot - xMinPlot) + xMinPlot
        noise := (rand.Float64()*3 - 1) * noiseMax
        yData[i] = trueSlope*xData[i] + trueIntercept + noise

        if yData[i] > trueSlope*xData[i]+trueIntercept {
            labels[i] = 1
        } else {
            labels[i] = 0
        }
    }
}

// 坐标转换：数学坐标转屏幕坐标
func mathToScreen(x, y float64) (float64, float64) {
    screenX := (x - xMinPlot) / (xMaxPlot - xMinPlot) * float64(screenWidth)
    screenY := (1 - (y - yMinPlot)/(yMaxPlot - yMinPlot)) * float64(screenHeight)
    return screenX, screenY
}

// 绘制网格
func drawGrid(screen *ebiten.Image) {
    // 垂直网格线
    for x := xMinPlot; x <= xMaxPlot; x += 2.0 {
        sx, _ := mathToScreen(x, 0)
        ebitenutil.DrawLine(screen, sx, 0, sx, float64(screenHeight), colorGrid)
    }
    // 水平网格线
    for y := yMinPlot; y <= yMaxPlot; y += 2.0 {
        _, sy := mathToScreen(0, y)
        ebitenutil.DrawLine(screen, 0, sy, float64(screenWidth), sy, colorGrid)
    }
}

// 绘制坐标轴及刻度
func drawAxis(screen *ebiten.Image) {
    // 绘制坐标轴
    x0, y0 := mathToScreen(0, 0)
    x1, _ := mathToScreen(xMaxPlot, 0)
    _, y1 := mathToScreen(0, yMaxPlot)

    ebitenutil.DrawLine(screen, x0, y0, x1, y0, colorText) // X轴
    ebitenutil.DrawLine(screen, x0, y0, x0, y1, colorText) // Y轴

    // 绘制刻度标签
    for x := xMinPlot; x <= xMaxPlot; x += 5.0 {
        sx, sy := mathToScreen(x, 0)
        ebitenutil.DrawLine(screen, sx, sy-5, sx, sy+5, colorText)
        text.Draw(screen, fmt.Sprintf("%.0f", x), basicfont.Face7x13, int(sx)-10, int(sy)+20, colorText)
    }

    for y := yMinPlot; y <= yMaxPlot; y += 5.0 {
        sx, sy := mathToScreen(0, y)
        ebitenutil.DrawLine(screen, sx-5, sy, sx+5, sy, colorText)
        text.Draw(screen, fmt.Sprintf("%.0f", y), basicfont.Face7x13, int(sx)-30, int(sy)+5, colorText)
    }

    // 绘制轴标签
    text.Draw(screen, "X", basicfont.Face7x13, int(x1)-20, int(y0)+20, colorText)
    text.Draw(screen, "Y", basicfont.Face7x13, int(x0)+10, int(y1)-10, colorText)
}

// 绘制数据点
func drawDataPoints(screen *ebiten.Image) {
    for i := 0; i < numPoints; i++ {
        sx, sy := mathToScreen(xData[i], yData[i])
        if labels[i] == 1 {
            ebitenutil.DrawCircle(screen, sx, sy, 2, colorClass1)
        } else {
            ebitenutil.DrawCircle(screen, sx, sy, 2, colorClass0)
        }
    }
}

// 绘制直线（真实边界和拟合边界）
func drawLine(screen *ebiten.Image, slope, intercept float64, clr color.Color) {
    x1, y1 := xMinPlot, slope*xMinPlot+intercept
    x2, y2 := xMaxPlot, slope*xMaxPlot+intercept

    sx1, sy1 := mathToScreen(x1, y1)
    sx2, sy2 := mathToScreen(x2, y2)

    ebitenutil.DrawLine(screen, sx1, sy1, sx2, sy2, clr)
}

// 绘制逻辑回归分类边界（a + b*x + c*y = 0）
func drawDecisionBoundary(screen *ebiten.Image, a, b, c float64, clr color.Color) {
    if c == 0 {
        return
    }
    x1 := xMinPlot
    y1 := (-a - b*x1) / c
    x2 := xMaxPlot
    y2 := (-a - b*x2) / c

    sx1, sy1 := mathToScreen(x1, y1)
    sx2, sy2 := mathToScreen(x2, y2)

    ebitenutil.DrawLine(screen, sx1, sy1, sx2, sy2, clr)
}

// 绘制方程文本：真实直线方程 + 拟合边界方程
func drawEquations(screen *ebiten.Image) {
    // 真实直线方程：y = trueSlope*x + trueIntercept
    trueEquation := fmt.Sprintf("True Line: y = %.4fx + %.4f", trueSlope, trueIntercept)
    text.Draw(screen, trueEquation, basicfont.Face7x13, 20, 40, colorText)

    // 拟合边界方程：a + b*x + c*y = 0 → 整理为 y = (-a - b*x)/c（c≠0时）
    if c != 0 {
        fitEquation := fmt.Sprintf("Fit Boundary: y = (%.4f + %.4fx) / (%.4f)", -a, -b, c)
        text.Draw(screen, fitEquation, basicfont.Face7x13, 20, 60, colorText)
    } else {
        // c=0 时特殊处理（实际训练中 c 一般不会长期为0，这里做兜底）
        fitEquation := fmt.Sprintf("Fit Boundary: %.4f + %.4fx = 0", a, b)
        text.Draw(screen, fitEquation, basicfont.Face7x13, 20, 60, colorText)
    }
}

// 绘制训练信息
func drawTrainingInfo(screen *ebiten.Image) {
    info := fmt.Sprintf("Training Progress\nIteration: %d/%d (%.2f%%)\nLoss: %.6f\nParameters: a=%.6f, b=%.6f, c=%.6f", 
        currentIter, totalIterations, float64(currentIter)/float64(totalIterations)*100, loss, a, b, c)
    text.Draw(screen, info, basicfont.Face7x13, 20, screenHeight-80, colorText)
}

// Game 实现 ebiten.Game 接口
type Game struct{}

func (g *Game) Update() error {
    if currentIter < totalIterations {
        gradA := gradientA(labels, xData, yData, a, b, c)
        gradB := gradientB(labels, xData, yData, a, b, c)
        gradC := gradientC(labels, xData, yData, a, b, c)

        a -= learningRate * gradA
        b -= learningRate * gradB
        c -= learningRate * gradC

        loss = crossEntropyLoss(labels, xData, yData, a, b, c)
        currentIter++
    }
    return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
    screen.Fill(colorBg)

    drawGrid(screen)
    drawAxis(screen)
    drawDataPoints(screen)

    drawLine(screen, trueSlope, trueIntercept, colorTrueLine)
    drawDecisionBoundary(screen, a, b, c, colorFitLine)

    drawEquations(screen)  // 新增：绘制方程文本
    drawTrainingInfo(screen)
}

func (g *Game) Layout(_, _ int) (int, int) {
    return screenWidth, screenHeight
}

func main() {
    generateData()
    ebiten.SetWindowSize(screenWidth, screenHeight)
    ebiten.SetWindowTitle("Logistic Regression Animation")
    ebiten.SetMaxTPS(60)

    if err := ebiten.RunGame(&Game{}); err != nil {
        panic(err)
    }
}