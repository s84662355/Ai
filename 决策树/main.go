package main

import (
	"fmt"
	"math"
	"sync"
)

// 数据集结构，包含特征名称、所有样本数据
type Dataset struct {
	FeatureNames []string            // 特征名称：年龄、服务时间、月消费、投诉次数、非保险费
	Samples      []map[string]string // 样本：每个样本是特征名到取值的映射，含"续签"结果
}

// 决策树节点结构
type TreeNode struct {
	Feature  string               // 当前节点依据的划分特征（如"月消费"）
	Children map[string]*TreeNode // 子节点，键是特征取值（如"低""中""高"）
	Class    string               // 叶节点的类别（如"续签""不续签"），非叶节点为空
}

// 计算数据集的信息熵
func calculateEntropy(dataset Dataset, resultFeature string) float64 {
	classCounts := make(map[string]int)
	for _, sample := range dataset.Samples {
		class := sample[resultFeature]
		classCounts[class]++
	}
	total := len(dataset.Samples)
	entropy := 0.0
	for _, count := range classCounts {
		p := float64(count) / float64(total)
		entropy -= p * math.Log2(p)
	}
	return entropy
}

// 按特征和取值划分数据集
func splitDataset(dataset Dataset, feature string, value string) Dataset {
	newSamples := make([]map[string]string, 0)
	for _, sample := range dataset.Samples {
		if sample[feature] == value {
			// 深拷贝样本（避免原数据被修改），并移除当前划分特征
			newSample := make(map[string]string)
			for k, v := range sample {
				if k != feature {
					newSample[k] = v
				}
			}
			newSamples = append(newSamples, newSample)
		}
	}
	// 保持特征名称，去掉已用的划分特征
	newFeatureNames := make([]string, 0)
	for _, fn := range dataset.FeatureNames {
		if fn != feature {
			newFeatureNames = append(newFeatureNames, fn)
		}
	}
	return Dataset{
		FeatureNames: newFeatureNames,
		Samples:      newSamples,
	}
}

// 计算某特征的信息增益
func calculateInfoGain(dataset Dataset, feature string, resultFeature string) float64 {
	baseEntropy := calculateEntropy(dataset, resultFeature)
	featureValues := make(map[string]bool)
	for _, sample := range dataset.Samples {
		featureValues[sample[feature]] = true
	}
	total := len(dataset.Samples)
	newEntropy := 0.0
	for value := range featureValues {
		subset := splitDataset(dataset, feature, value)
		p := float64(len(subset.Samples)) / float64(total)
		newEntropy += p * calculateEntropy(subset, resultFeature)
	}
	return baseEntropy - newEntropy
}

// 选择最优划分特征（信息增益最大）
func chooseBestFeature(dataset Dataset, resultFeature string) string {
	maxGain := -1.0
	bestFeature := ""
	for _, feature := range dataset.FeatureNames {
		if feature == resultFeature {
			continue
		}
		gain := calculateInfoGain(dataset, feature, resultFeature)
		if gain > maxGain {
			maxGain = gain
			bestFeature = feature
		}
	}
	return bestFeature
}

// 检查数据集是否全属于同一类别
func isPure(dataset Dataset, resultFeature string) (bool, string) {
	firstClass := dataset.Samples[0][resultFeature]
	for _, sample := range dataset.Samples {
		if sample[resultFeature] != firstClass {
			return false, ""
		}
	}
	return true, firstClass
}

// 递归构建决策树
func buildDecisionTree(dataset Dataset, resultFeature string) *TreeNode {
	// 检查是否纯类别
	isPureSet, class := isPure(dataset, resultFeature)
	if isPureSet {
		return &TreeNode{
			Class: class,
		}
	}
	// 无可用特征（理论上不会到这，因为纯类别已提前判断）
	if len(dataset.FeatureNames) == 0 {
		return &TreeNode{
			Class: class,
		}
	}
	// 选最优特征
	bestFeature := chooseBestFeature(dataset, resultFeature)
	tree := &TreeNode{
		Feature:  bestFeature,
		Children: make(map[string]*TreeNode),
		Class:    "",
	}
	// 获取该特征所有可能取值
	featureValues := make(map[string]bool)
	for _, sample := range dataset.Samples {
		featureValues[sample[bestFeature]] = true
	}
	// 递归构建子树
	var wg sync.WaitGroup
	for value := range featureValues {
		wg.Add(1)
		go func(v string) {
			defer wg.Done()
			subset := splitDataset(dataset, bestFeature, v)
			tree.Children[v] = buildDecisionTree(subset, resultFeature)
		}(value)
	}
	wg.Wait()
	return tree
}

// 打印决策树（缩进展示结构）
func printTree(node *TreeNode, indent string) {
	if node.Class != "" {
		fmt.Printf("%s└── 类别: %s\n", indent, node.Class)
		return
	}
	fmt.Printf("%s└── 特征: %s\n", indent, node.Feature)
	for value, child := range node.Children {
		fmt.Printf("%s    ├── 取值: %s\n", indent, value)
		printTree(child, indent+"    ")
	}
}

func main() {
	// 初始化数据集
	featureNames := []string{"年龄", "服务时间", "月消费", "投诉次数", "非保险费", "续签"}
	samples := []map[string]string{
		{"序号": "1", "年龄": "青年", "服务时间": "短", "月消费": "低", "投诉次数": "0", "非保险费": "基础", "续签": "不续签"},
		{"序号": "2", "年龄": "青年", "服务时间": "中", "月消费": "中", "投诉次数": "1", "非保险费": "基础", "续签": "不续签"},
		{"序号": "3", "年龄": "中年", "服务时间": "长", "月消费": "高", "投诉次数": "0", "非保险费": "高级", "续签": "续签"},
		{"序号": "4", "年龄": "老年", "服务时间": "长", "月消费": "中", "投诉次数": "0", "非保险费": "高级", "续签": "续签"},
		{"序号": "5", "年龄": "老年", "服务时间": "长", "月消费": "低", "投诉次数": "≥2", "非保险费": "基础", "续签": "不续签"},
		{"序号": "6", "年龄": "中年", "服务时间": "中", "月消费": "高", "投诉次数": "1", "非保险费": "高级", "续签": "续签"},
		{"序号": "7", "年龄": "青年", "服务时间": "短", "月消费": "中", "投诉次数": "0", "非保险费": "基础", "续签": "不续签"},
		{"序号": "8", "年龄": "中年", "服务时间": "长", "月消费": "高", "投诉次数": "0", "非保险费": "基础", "续签": "续签"},
		{"序号": "9", "年龄": "老年", "服务时间": "中", "月消费": "中", "投诉次数": "1", "非保险费": "高级", "续签": "续签"},
		{"序号": "10", "年龄": "青年", "服务时间": "长", "月消费": "高", "投诉次数": "≥2", "非保险费": "高级", "续签": "不续签"},
		{"序号": "11", "年龄": "中年", "服务时间": "短", "月消费": "高", "投诉次数": "0", "非保险费": "高级", "续签": "续签"},
		{"序号": "12", "年龄": "老年", "服务时间": "短", "月消费": "低", "投诉次数": "0", "非保险费": "基础", "续签": "不续签"},
		{"序号": "13", "年龄": "青年", "服务时间": "中", "月消费": "高", "投诉次数": "1", "非保险费": "高级", "续签": "续签"},
		{"序号": "14", "年龄": "中年", "服务时间": "中", "月消费": "中", "投诉次数": "0", "非保险费": "基础", "续签": "不续签"},
		{"序号": "15", "年龄": "老年", "服务时间": "长", "月消费": "高", "投诉次数": "≥2", "非保险费": "高级", "续签": "不续签"},
		{"序号": "16", "年龄": "青年", "服务时间": "短", "月消费": "低", "投诉次数": "0", "非保险费": "基础", "续签": "不续签"},
		{"序号": "17", "年龄": "中年", "服务时间": "中", "月消费": "高", "投诉次数": "0", "非保险费": "基础", "续签": "续签"},
		{"序号": "18", "年龄": "老年", "服务时间": "中", "月消费": "高", "投诉次数": "1", "非保险费": "高级", "续签": "续签"},
		{"序号": "19", "年龄": "青年", "服务时间": "长", "月消费": "中", "投诉次数": "0", "非保险费": "高级", "续签": "续签"},
		{"序号": "20", "年龄": "中年", "服务时间": "长", "月消费": "中", "投诉次数": "≥2", "非保险费": "基础", "续签": "不续签"},
	}
	dataset := Dataset{
		FeatureNames: featureNames,
		Samples:      samples,
	}
	resultFeature := "续签"

	// 构建决策树
	root := buildDecisionTree(dataset, resultFeature)

	// 打印决策树
	fmt.Println("决策树结构:")
	printTree(root, "")

	// 示例：用决策树预测新样本（可扩展为批量预测）
	newSample := map[string]string{
		"年龄":   "中年",
		"服务时间": "长",
		"月消费":  "高",
		"投诉次数": "0",
		"非保险费": "高级",
	}
	fmt.Printf("\n预测新样本: %+v\n", newSample)
	currentNode := root
	for currentNode.Class == "" {
		featureValue := newSample[currentNode.Feature]
		currentNode = currentNode.Children[featureValue]
		if currentNode == nil {
			fmt.Println("无法预测（特征取值未见过）")
			return
		}
	}
	fmt.Printf("预测结果: %s\n", currentNode.Class)
}
