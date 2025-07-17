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
	return math.Sqrt(math.Pow(p1.X-p2.X, 2) + math.Pow(p1.Y-p2.Y, 2))
}

// 生成随机点
func generateRandomPoints(count int, min, max float64) []Point {
	points := make([]Point, count)
	for i := 0; i < count; i++ {
		points[i] = Point{
			X: min + rand.Float64()*(max-min),
			Y: min + rand.Float64()*(max-min),
		}
	}
	return points
}

// 可视化窗口和动画逻辑
type Game struct {
	points        []Point       // 所有数据点
	clusters      []int         // 当前聚类结果
	centroids     []Point       // 当前聚类中心
	prevCentroids []Point       // 上一轮聚类中心（用于动画过渡）
	k             int           // 聚类数量
	width         int           // 窗口宽度
	height        int           // 窗口高度
	iteration     int           // 当前迭代次数
	animProgress  float64       // 动画进度（0-1）
	animSpeed     float64       // 动画速度
	converged     bool          // 是否收敛
}

func NewGame(points []Point, k int) *Game {
	rand.Seed(time.Now().UnixNano())
	
	// 初始化聚类中心
	centroids := make([]Point, k)
	for i := range centroids {
		centroids[i] = points[rand.Intn(len(points))]
	}
	
	return &Game{
		points:        points,
		clusters:      make([]int, len(points)),
		centroids:     centroids,
		prevCentroids: make([]Point, k),
		k:             k,
		width:         800,
		height:        600,
		iteration:     0,
		animProgress:  0,
		animSpeed:     0.01, // 每次更新的动画进度增量
		converged:     false,
	}
}

func (g *Game) Layout(outsideWidth, outsideHeight int) (int, int) {
	return g.width, g.height
}

// 执行一次K-means迭代
func (g *Game) stepKmeans() bool {
	// 保存当前中心作为上一轮中心（用于动画过渡）
	copy(g.prevCentroids, g.centroids)
	
	changed := false
	
	// 1. 分配每个点到最近的聚类中心
	for i, p := range g.points {
		minDist := math.MaxFloat64
		closest := g.clusters[i]
		
		for j, c := range g.centroids {
			dist := distance(p, c)
			if dist < minDist {
				minDist = dist
				closest = j
			}
		}
		
		if closest != g.clusters[i] {
			g.clusters[i] = closest
			changed = true
		}
	}
	
	// 2. 更新聚类中心为每个聚类的平均值
	newCentroids := make([]Point, g.k)
	counts := make([]int, g.k)
	
	for i, c := range g.clusters {
		newCentroids[c].X += g.points[i].X
		newCentroids[c].Y += g.points[i].Y
		counts[c]++
	}
	
	for j := 0; j < g.k; j++ {
		if counts[j] > 0 {
			newCentroids[j].X /= float64(counts[j])
			newCentroids[j].Y /= float64(counts[j])
		}
	}
	
	g.centroids = newCentroids
	g.iteration++
	
	return changed
}

func (g *Game) Update() error {
	if g.converged {
		return nil
	}
	
	// 动画进行中，更新进度
	if g.animProgress < 1.0 {
		g.animProgress += g.animSpeed
		if g.animProgress > 1.0 {
			g.animProgress = 1.0
		}
		return nil
	}
	
	// 动画完成，执行下一步K-means迭代
	changed := g.stepKmeans()
	if !changed {
		g.converged = true
	}
	g.animProgress = 0 // 重置动画进度
	
	return nil
}

func (g *Game) Draw(screen *ebiten.Image) {
	// 填充背景为白色
	screen.Fill(color.White)

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
	}

	// 绘制所有点（按聚类颜色区分）
	for i, p := range g.points {
		clusterID := g.clusters[i]
		c := colors[clusterID%len(colors)]

		// 坐标映射到窗口尺寸
		x := int(p.X * float64(g.width) / 100)
		y := int(p.Y * float64(g.height) / 100)

		// 绘制点（4x4的方块）
		for dx := -2; dx <= 2; dx++ {
			for dy := -2; dy <= 2; dy++ {
				screen.Set(x+dx, y+dy, c)
			}
		}
	}

	// 绘制聚类中心（带动画过渡效果）
	for i := 0; i < g.k; i++ {
		// 计算动画过渡中的中心位置
		x := g.prevCentroids[i].X + (g.centroids[i].X-g.prevCentroids[i].X)*g.animProgress
		y := g.prevCentroids[i].Y + (g.centroids[i].Y-g.prevCentroids[i].Y)*g.animProgress
		
		// 映射到窗口坐标
		screenX := int(x * float64(g.width) / 100)
		screenY := int(y * float64(g.height) / 100)

		// 绘制中心（8x8的黑色方块）
		for dx := -4; dx <= 4; dx++ {
			for dy := -4; dy <= 4; dy++ {
				screen.Set(screenX+dx, screenY+dy, color.Black)
			}
		}
	}

	// 显示迭代信息
	status := fmt.Sprintf("K-means 聚类动画 (k=%d) - 迭代次数: %d", g.k, g.iteration)
	if g.converged {
		status += " - 已收敛！"
	}
	ebitenutil.DebugPrint(screen, status)
}

func main() {
	// 生成100个随机点（范围0-100）
	points := generateRandomPoints(300, 0, 100)
	
	// 聚类数量
	k :=5
	
	// 初始化游戏（包含动画逻辑）
	game := NewGame(points, k)
	ebiten.SetWindowSize(game.width, game.height)
	ebiten.SetWindowTitle("K-means 聚类过程动画")

	// 运行动画
	if err := ebiten.RunGame(game); err != nil {
		panic(err)
	}
}