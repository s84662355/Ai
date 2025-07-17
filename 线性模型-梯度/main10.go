package main

import (
	"fmt"
	"image/color"
	"math"
	"math/rand"
	"time"

	"github.com/hajimehoshi/ebiten/v2"
	"github.com/hajimehoshi/ebiten/v2/ebitenutil"
)

// 数据点结构
type Point struct {
	X, Y float64
}

// 计算欧氏距离
func distance(p1, p2 Point) float64 {
	return math.Hypot(p1.X-p2.X, p1.Y-p2.Y)
}

// 生成带聚类特性的随机点（便于展示均值漂移效果）
func generateClusteredPoints(total int, clusters int) []Point {
	rand.Seed(time.Now().UnixNano())
	points := make([]Point, 0, total)
	
	// 生成几个密集聚类中心
	centers := make([]Point, clusters)
	for i := 0; i < clusters; i++ {
		centers[i] = Point{
			X: 20 + rand.Float64()*60, // 范围20-80
			Y: 20 + rand.Float64()*60,
		}
	}
	
	// 围绕每个中心生成点
	pointsPerCluster := total / clusters
	for _, c := range centers {
		for i := 0; i < pointsPerCluster; i++ {
			// 添加高斯分布的噪声
			points = append(points, Point{
				X: c.X + (rand.Float64()*2-1)*8, // 标准差8
				Y: c.Y + (rand.Float64()*2-1)*8,
			})
		}
	}
	
	// 补充剩余点
	for len(points) < total {
		points = append(points, Point{
			X: 10 + rand.Float64()*80,
			Y: 10 + rand.Float64()*80,
		})
	}
	
	return points
}

// 均值漂移算法结构体（包含动画状态）
type MeanShift struct {
	points       []Point       // 原始数据点
	modes        []Point       // 每个点的漂移终点（模式点）
	currentModes []Point       // 当前漂移位置（用于动画）
	labels       []int         // 聚类标签
	bandwidth    float64       // 带宽（核函数半径）
	iterations   int           // 总迭代次数
	currentStep  int           // 当前动画步骤
	converged    bool          // 是否收敛
}

func NewMeanShift(points []Point, bandwidth float64) *MeanShift {
	ms := &MeanShift{
		points:       points,
		modes:        make([]Point, len(points)),
		currentModes: make([]Point, len(points)),
		labels:       make([]int, len(points)),
		bandwidth:    bandwidth,
		iterations:   0,
		currentStep:  0,
		converged:    false,
	}
	
	// 初始化模式点为原始点（漂移起点）
	copy(ms.modes, points)
	copy(ms.currentModes, points)
	
	return ms
}

// 执行一步均值漂移计算
func (ms *MeanShift) Step() bool {
	converged := true
	bandwidthSq := ms.bandwidth * ms.bandwidth // 带宽平方（优化计算）
	
	// 对每个点执行一次漂移计算
	for i := range ms.modes {
		currentMode := ms.modes[i]
		sumX, sumY := 0.0, 0.0
		totalWeight := 0.0
		
		// 计算带宽范围内的加权平均
		for _, p := range ms.points {
			distSq := (p.X-currentMode.X)*(p.X-currentMode.X) + (p.Y-currentMode.Y)*(p.Y-currentMode.Y)
			if distSq <= bandwidthSq {
				// 高斯核函数权重
				weight := math.Exp(-distSq / (2 * bandwidthSq))
				sumX += p.X * weight
				sumY += p.Y * weight
				totalWeight += weight
			}
		}
		
		// 计算新的模式点
		if totalWeight > 0 {
			newMode := Point{
				X: sumX / totalWeight,
				Y: sumY / totalWeight,
			}
			
			// 检查是否收敛（移动距离小于阈值）
			if distance(newMode, currentMode) > 0.01 {
				ms.modes[i] = newMode
				converged = false
			}
		}
	}
	
	ms.iterations++
	return converged
}

// 动画更新当前显示的模式点（平滑过渡）
func (ms *MeanShift) UpdateAnimation(progress float64) {
	for i := range ms.currentModes {
		// 线性插值实现平滑动画
		ms.currentModes[i].X = ms.points[i].X + (ms.modes[i].X-ms.points[i].X)*progress
		ms.currentModes[i].Y = ms.points[i].Y + (ms.modes[i].Y-ms.points[i].Y)*progress
	}
	
	// 收敛后计算聚类标签（合并相似的模式点）
	if ms.converged {
		clusterID := 0
		clusterCenters := make([]Point, 0)
		
		for i := range ms.labels {
			found := false
			// 检查是否与已有聚类中心相似
			for j, center := range clusterCenters {
				if distance(ms.modes[i], center) < ms.bandwidth/2 {
					ms.labels[i] = j
					found = true
					break
				}
			}
			if !found {
				clusterCenters = append(clusterCenters, ms.modes[i])
				ms.labels[i] = clusterID
				clusterID++
			}
		}
	}
}

// 可视化窗口
type Game struct {
	ms        *MeanShift
	width     int
	height    int
	animSpeed float64
	animProg  float64
}

func NewGame(points []Point, bandwidth float64) *Game {
	return &Game{
		ms:        NewMeanShift(points, bandwidth),
		width:     800,
		height:    600,
		animSpeed: 0.03,
		animProg:  0,
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return g.width, g.height
}

func (g *Game) Update() error {
	if g.ms.converged {
		return nil
	}
	
	// 动画进度更新
	g.animProg += g.animSpeed
	if g.animProg >= 1.0 {
		// 动画结束，执行下一步均值漂移
		g.animProg = 0
		g.ms.converged = g.ms.Step()
		g.ms.currentStep++
	}
	
	// 更新当前动画帧的显示状态
	g.ms.UpdateAnimation(g.animProg)
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// 白色背景
	screen.Fill(color.White)

	// 定义聚类颜色
	// 定义聚类颜色
	colors := []color.Color{
		color.RGBA{255, 0, 0, 255},    // 红色
		color.RGBA{0, 255, 0, 255},    // 绿色
		color.RGBA{0, 0, 255, 255},    // 蓝色
		color.RGBA{255, 165, 0, 255},  // 橙色
		color.RGBA{128, 0, 128, 255},  // 紫色
		color.RGBA{0, 255, 255, 255},  // 青色
		color.RGBA{255, 255, 0, 255},  // 黄色
		color.RGBA{128, 128, 128, 255}, // 灰色
		color.RGBA{18, 218, 18, 255},  
		color.RGBA{181, 28, 8, 255}, 
		color.RGBA{81, 32, 48, 255},  
		color.RGBA{231,54, 88, 255},  
	}

	// 绘制原始数据点（灰色小点点）
	for _, p := range g.ms.points {
		x := int(p.X * float64(g.width) / 100)
		y := int(p.Y * float64(g.height) / 100)
		for dx := -1; dx <= 1; dx++ {
			for dy := -1; dy <= 1; dy++ {
				screen.Set(x+dx, y+dy, color.Gray{Y: 200})
			}
		}
	}

	// 绘制漂移轨迹线（浅色）
	for i := range g.ms.points {
		start := g.ms.points[i]
		current := g.ms.currentModes[i]
		startX := int(start.X * float64(g.width) / 100)
		startY := int(start.Y * float64(g.height) / 100)
		currentX := int(current.X * float64(g.width) / 100)
		currentY := int(current.Y * float64(g.height) / 100)
		drawLine(screen, startX, startY, currentX, currentY, color.Gray{Y: 150})
	}

	// 绘制当前模式点（带聚类颜色）
	for i, m := range g.ms.currentModes {
		var c color.Color
		if g.ms.converged {
			// 收敛后按聚类着色
			c = colors[g.ms.labels[i]%len(colors)]
		} else {
			// 收敛前用统一颜色
			c = color.RGBA{0, 0, 255, 200}
		}
		
		x := int(m.X * float64(g.width) / 100)
		y := int(m.Y * float64(g.height) / 100)
		for dx := -2; dx <= 2; dx++ {
			for dy := -2; dy <= 2; dy++ {
				screen.Set(x+dx, y+dy, c)
			}
		}
	}

	// 显示算法状态
	status := fmt.Sprintf("均值漂移聚类 - 迭代: %d, 带宽: %.1f", g.ms.iterations, g.ms.bandwidth)
	if g.ms.converged {
		status += " - 已收敛！"
	}
	ebitenutil.DebugPrint(screen, status)
}

// 绘制线段的辅助函数
func drawLine(screen *ebiten.Image, x0, y0, x1, y1 int, c color.Color) {
	dx := abs(x1 - x0)
	dy := abs(y1 - y0)
	sx, sy := 1, 1
	if x0 > x1 {
		sx = -1
	}
	if y0 > y1 {
		sy = -1
	}
	err := dx - dy

	for {
		screen.Set(x0, y0, c)
		if x0 == x1 && y0 == y1 {
			break
		}
		e2 := 2 * err
		if e2 > -dy {
			err -= dy
			x0 += sx
		}
		if e2 < dx {
			err += dx
			y0 += sy
		}
	}
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func main() {
	// 生成100个带聚类特性的点（3个自然聚类）
	points := generateClusteredPoints(600, 5)
	
	// 初始化均值漂移（带宽设为8.0，控制聚类粒度）
	game := NewGame(points, 5)
	ebiten.SetWindowSize(game.width, game.height)
	ebiten.SetWindowTitle("均值漂移聚类动画")

	// 运行动画
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}