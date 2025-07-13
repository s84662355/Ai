package main

import (
	"image/color"
 
	"time"
	"os/exec"

 "fmt"
	"github.com/pa-m/sklearn/datasets"
	"github.com/pa-m/sklearn/linear_model"
 
	"gonum.org/v1/gonum/mat"
	"gonum.org/v1/plot"
	"gonum.org/v1/plot/plotter"
	"gonum.org/v1/plot/vg"
)

func main() {
	// Load the diabetes dataset
	diabetes := datasets.LoadDiabetes()

	// Use only one feature (slice the 3rd feature, index 2)
	NSamples, _ := diabetes.X.Dims()
 
	diabetesX := diabetes.X.Slice(0, NSamples, 0, 1).(*mat.Dense)

 

	// Split the data into training/testing sets (last 20 samples as test set)
	diabetesXtrain := diabetesX.Slice(0, NSamples-20, 0, 1).(*mat.Dense)
	diabetesXtest := diabetesX.Slice(NSamples-20, NSamples, 0,1).(*mat.Dense)

	// Split the targets into training/testing sets
	diabetesYtrain := diabetes.Y.Slice(0, NSamples-20, 0, 1).(*mat.Dense)
	diabetesYtest := diabetes.Y.Slice(NSamples-20, NSamples, 0, 1).(*mat.Dense)

	// Create linear regression object
 	///regr := linearmodel.NewLinearRegression()


    regr := linearmodel.NewSGDRegressor()
   regr.NJobs                = 10000    // 最大迭代次数
    regr.Alpha = 0.001       // 初始学习率


	// Train the model using the training sets
	regr.Fit(diabetesXtrain, diabetesYtrain)

	// Make predictions using the testing set
	NTestSamples := 20
	diabetesYpred := mat.NewDense(NTestSamples, 1, nil)
	regr.Predict(diabetesXtest, diabetesYpred)

	// Print coefficients
	fmt.Printf("Coefficients: %.8f\n", mat.Formatted(regr.Coef))
	fmt.Printf("Coefficients: %.8f\n",  regr.Coef) 
	fmt.Printf("Coefficients: %.8f\n",  regr.Intercept) 

 

 

	// Plot outputs (set canPlot to true to enable)
	canPlot := true // 改为true启用可视化
	if canPlot {
		// Create a new plot
		p  := plot.New()
	 

		p.Title.Text = "Diabetes Dataset Linear Regression"
		p.X.Label.Text = "Feature (Scaled)"
		p.Y.Label.Text = "Target"

		// Convert matrix data to plotter.XYs format
		xys := func(X, Y mat.Matrix) plotter.XYs {
			var data plotter.XYs
			n, _ := X.Dims()
			for i := 0; i < n; i++ {
				data = append(data, plotter.XY{
					X: X.At(i, 0),
					Y: Y.At(i, 0),
				})
			}
			return data
		}

		// Add test data scatter points
		scatter, err := plotter.NewScatter(xys(diabetesXtest, diabetesYtest))
		if err != nil {
			panic(err)
		}
		scatter.Color = color.RGBA{255, 0, 0, 255} // 红色散点（测试数据）
		p.Add(scatter)

		// Add regression line
		line, err := plotter.NewLine(xys(diabetesXtest, diabetesYpred))
		if err != nil {
			panic(err)
		}
		line.Color = color.RGBA{0, 0, 255, 255} // 蓝色线（预测结果）
		p.Add(line)

		// Save plot to PNG file
		pngFile := "linear_regression_diabetes.png"
		if err := p.Save(8*vg.Inch, 6*vg.Inch, pngFile); err != nil {
			panic(err)
		}
		fmt.Printf("Plot saved to %s\n", pngFile)

		// Optional: Open the image with default viewer
		cmd := exec.Command("cmd", "/c", "start", pngFile) // Windows
		// cmd := exec.Command("open", pngFile) // macOS
		// cmd := exec.Command("xdg-open", pngFile) // Linux
		if err := cmd.Run(); err != nil {
			fmt.Printf("Warning: Could not open image viewer: %v\n", err)
		}

		// Wait to ensure the file is opened before program exits
		time.Sleep(2 * time.Second)
	}
}