package main

import (
	"bufio"
	"encoding/binary"
	"fmt"
	"io"
	"math"
	"os"
	"runtime"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// FilterConfig holds all configuration parameters for filtering
type FilterConfig struct {
	// Neighbor filtering
	EnableNeighborFilter bool
	Radius              float32
	MinNeighbors        int
	
	// Surface area filtering
	EnableAreaFilter    bool
	MaxArea             float32
	
	// Output options
	SaveDiscarded       bool
	OutputDir          string
}

// Point represents a 3D point
type Point struct {
	X, Y, Z float32
}

// Splat structure matching the PLY format
type Splat struct {
	X, Y, Z          float32 // Position
	NX, NY, NZ       float32 // Normal
	FDC0, FDC1, FDC2 float32 // DC components
	FRest            [45]float32 // Rest of features
	Opacity          float32
	Scale0, Scale1, Scale2 float32 // Scale
	Rot0, Rot1, Rot2, Rot3 float32 // Rotation quaternion
}

// FilterResult holds statistics about the filtering process
type FilterResult struct {
	TotalSplats        int
	KeptSplats         int
	DiscardedByArea    int
	DiscardedByNeighbors int
	DiscardedByBoth    int
}

// ColmapPoint matches the binary format of points3D.bin
type ColmapPoint struct {
	ID          uint64
	X, Y, Z     float64
	R, G, B     uint8
	Error       float64
	TrackLength uint64
}

// KDTree node structure
type KDNode struct {
	Point       Point
	Left, Right *KDNode
	Axis        int
}

// Cell represents a spatial cell in the grid
type Cell struct {
	X, Y, S int
}

// GetBBox returns the bounding box of the cell with optional margin
func (c *Cell) GetBBox(margin float32) (minX, minY, maxX, maxY float32) {
	minX = float32(c.X * c.S) - margin
	minY = float32(c.Y * c.S) - margin
	maxX = minX + float32(c.S) + 2*margin
	maxY = minY + float32(c.S) + 2*margin
	return
}

// ParseCellID parses a cell ID string like "0x-1y5s" into a Cell struct
func ParseCellID(id string) (*Cell, error) {
	var x, y, s int
	_, err := fmt.Sscanf(id, "%dx%dy%ds", &x, &y, &s)
	if err != nil {
		return nil, fmt.Errorf("invalid cell ID format: %v", err)
	}
	return &Cell{X: x, Y: y, S: s}, nil
}

// Build KD-tree recursively
func buildKDTree(points []Point, depth int) *KDNode {
	if len(points) == 0 {
		return nil
	}

	axis := depth % 3
	mid := len(points) / 2

	// Sort points based on the current axis
	switch axis {
	case 0: // X axis
		quickSortByAxis(points, 0, len(points)-1, func(p Point) float32 { return p.X })
	case 1: // Y axis
		quickSortByAxis(points, 0, len(points)-1, func(p Point) float32 { return p.Y })
	case 2: // Z axis
		quickSortByAxis(points, 0, len(points)-1, func(p Point) float32 { return p.Z })
	}

	node := &KDNode{
		Point: points[mid],
		Axis:  axis,
	}

	// Recursively build left and right subtrees
	if mid > 0 {
		node.Left = buildKDTree(points[:mid], depth+1)
	}
	if mid+1 < len(points) {
		node.Right = buildKDTree(points[mid+1:], depth+1)
	}

	return node
}

// Quick sort implementation for points based on a given axis
func quickSortByAxis(points []Point, low, high int, getValue func(Point) float32) {
	if low < high {
		pivot := partition(points, low, high, getValue)
		quickSortByAxis(points, low, pivot-1, getValue)
		quickSortByAxis(points, pivot+1, high, getValue)
	}
}

func partition(points []Point, low, high int, getValue func(Point) float32) int {
	pivot := getValue(points[high])
	i := low - 1

	for j := low; j < high; j++ {
		if getValue(points[j]) <= pivot {
			i++
			points[i], points[j] = points[j], points[i]
		}
	}

	points[i+1], points[high] = points[high], points[i+1]
	return i + 1
}

// getBoundingBox returns min/max coordinates of points
func getBoundingBox(points []Point) (minX, minY, minZ, maxX, maxY, maxZ float32) {
	if len(points) == 0 {
		return
	}
	minX, minY, minZ = points[0].X, points[0].Y, points[0].Z
	maxX, maxY, maxZ = minX, minY, minZ
	
	for _, p := range points {
		if p.X < minX { minX = p.X }
		if p.Y < minY { minY = p.Y }
		if p.Z < minZ { minZ = p.Z }
		if p.X > maxX { maxX = p.X }
		if p.Y > maxY { maxY = p.Y }
		if p.Z > maxZ { maxZ = p.Z }
	}
	return
}

// filterPointsByBounds removes points outside the bounding box + radius
func filterPointsByBounds(points []Point, splats []Point, radius float32) []Point {
	// Get bounding box of splats
	minX, minY, minZ, maxX, maxY, maxZ := getBoundingBox(splats)
	
	// Expand by radius
	minX -= radius
	minY -= radius
	minZ -= radius
	maxX += radius
	maxY += radius
	maxZ += radius
	
	// Filter points
	filtered := make([]Point, 0, len(points))
	for _, p := range points {
		if p.X >= minX && p.X <= maxX &&
		   p.Y >= minY && p.Y <= maxY &&
		   p.Z >= minZ && p.Z <= maxZ {
			filtered = append(filtered, p)
		}
	}
	return filtered
}

// countPointsInRadius now has early termination
func countPointsInRadius(root *KDNode, point Point, radiusSq float32, minCount int) int {
	if root == nil {
		return 0
	}

	count := 0
	countPointsInRadiusRecursive(root, point, radiusSq, &count, minCount)
	return count
}

func countPointsInRadiusRecursive(node *KDNode, point Point, radiusSq float32, count *int, minCount int) {
	if node == nil || *count >= minCount {
		return
	}

	dx := point.X - node.Point.X
	dy := point.Y - node.Point.Y
	dz := point.Z - node.Point.Z
	distSq := dx*dx + dy*dy + dz*dz

	if distSq <= radiusSq {
		*count++
		if *count >= minCount {
			return
		}
	}

	var axisDist float32
	switch node.Axis {
	case 0:
		axisDist = dx
	case 1:
		axisDist = dy
	case 2:
		axisDist = dz
	}

	if axisDist <= 0 {
		countPointsInRadiusRecursive(node.Left, point, radiusSq, count, minCount)
		if *count < minCount && axisDist*axisDist <= radiusSq {
			countPointsInRadiusRecursive(node.Right, point, radiusSq, count, minCount)
		}
	} else {
		countPointsInRadiusRecursive(node.Right, point, radiusSq, count, minCount)
		if *count < minCount && axisDist*axisDist <= radiusSq {
			countPointsInRadiusRecursive(node.Left, point, radiusSq, count, minCount)
		}
	}
}

func readPoints3DBin(filename string) ([]Point, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, fmt.Errorf("opening points3D.bin: %v", err)
	}
	defer file.Close()

	// Read number of points (uint64)
	var numPoints uint64
	if err := binary.Read(file, binary.LittleEndian, &numPoints); err != nil {
		return nil, fmt.Errorf("reading point count: %v", err)
	}

	points := make([]Point, 0, numPoints)
	point := ColmapPoint{}

	for i := uint64(0); i < numPoints; i++ {
		if err := binary.Read(file, binary.LittleEndian, &point); err != nil {
			return nil, fmt.Errorf("reading point %d: %v", i, err)
		}
		points = append(points, Point{
			X: float32(point.X),
			Y: float32(point.Y),
			Z: float32(point.Z),
		})
	}

	return points, nil
}

func readPLYHeader(file *os.File) (vertexCount int, dataOffset int64, err error) {
	scanner := bufio.NewScanner(file)
	headerBytes := int64(0)

	for scanner.Scan() {
		line := scanner.Text()
		headerBytes += int64(len(line) + 1)

		if strings.Contains(line, "element vertex") {
			fmt.Sscanf(line, "element vertex %d", &vertexCount)
		} else if line == "end_header" {
			return vertexCount, headerBytes, nil
		}
	}
	return 0, 0, fmt.Errorf("invalid PLY header")
}

func readSplats(filename string) ([]Point, []Splat, error) {
	file, err := os.Open(filename)
	if err != nil {
		return nil, nil, fmt.Errorf("opening splat file: %v", err)
	}
	defer file.Close()

	vertexCount, dataOffset, err := readPLYHeader(file)
	if err != nil {
		return nil, nil, err
	}

	splats := make([]Splat, vertexCount)
	if _, err := file.Seek(dataOffset, io.SeekStart); err != nil {
		return nil, nil, err
	}

	if err := binary.Read(file, binary.LittleEndian, &splats); err != nil {
		return nil, nil, err
	}

	points := make([]Point, vertexCount)
	for i, s := range splats {
		points[i] = Point{X: s.X, Y: s.Y, Z: s.Z}
	}

	return points, splats, nil
}

func filterSplats(points []Point, splats []Splat, kdRoot *KDNode, config FilterConfig) ([]int, []int, FilterResult) {
	numSplats := len(splats)
	results := make([]bool, numSplats)
	var progress int64
	var wg sync.WaitGroup
	
	// Process in parallel
	numCPU := runtime.NumCPU()
	chunkSize := int(math.Max(1000, float64(numSplats)/float64(numCPU*2)))
	radiusSq := config.Radius * config.Radius
	
	processChunk := func(start, end int) {
		defer wg.Done()
		
		for i := start; i < end; i++ {
			// Initialize as kept
			keep := true
			
			// Check neighbors if enabled
			hasEnoughNeighbors := true
			if config.EnableNeighborFilter {
				count := countPointsInRadius(kdRoot, points[i], radiusSq, config.MinNeighbors)
				hasEnoughNeighbors = count >= config.MinNeighbors
				keep = keep && hasEnoughNeighbors
			}
			
			// Check area if enabled
			isAreaOk := true
			if config.EnableAreaFilter {
				area := computeSplatArea(&splats[i])
				isAreaOk = area <= config.MaxArea
				keep = keep && isAreaOk
			}
			
			results[i] = keep
			
			if atomic.AddInt64(&progress, 1)%10000 == 0 {
				fmt.Printf("\rProgress: %.1f%%", float64(progress)*100/float64(numSplats))
			}
		}
	}
	
	// Start timing the parallel processing
	filterStart := time.Now()
	
	for start := 0; start < numSplats; start += chunkSize {
		end := start + chunkSize
		if end > numSplats {
			end = numSplats
		}
		wg.Add(1)
		go processChunk(start, end)
	}
	wg.Wait()
	
	filterTime := time.Since(filterStart)
	fmt.Printf("\nFiltering completed in %.1f seconds\n", filterTime.Seconds())
	
	// Collect results and statistics
	collectStart := time.Now()
	var keptIndices, discardedIndices []int
	stats := FilterResult{TotalSplats: numSplats}
	
	for i := 0; i < numSplats; i++ {
		if results[i] {
			keptIndices = append(keptIndices, i)
			stats.KeptSplats++
		} else {
			if config.SaveDiscarded {
				discardedIndices = append(discardedIndices, i)
			}
			
			// Count reason for discarding
			hasEnoughNeighbors := true
			isAreaOk := true
			
			if config.EnableNeighborFilter {
				count := countPointsInRadius(kdRoot, points[i], radiusSq, config.MinNeighbors)
				hasEnoughNeighbors = count >= config.MinNeighbors
			}
			
			if config.EnableAreaFilter {
				area := computeSplatArea(&splats[i])
				isAreaOk = area <= config.MaxArea
			}
			
			if !hasEnoughNeighbors && !isAreaOk {
				stats.DiscardedByBoth++
			} else if !hasEnoughNeighbors {
				stats.DiscardedByNeighbors++
			} else {
				stats.DiscardedByArea++
			}
		}
	}
	
	collectTime := time.Since(collectStart)
	fmt.Printf("Results collection completed in %.1f seconds\n", collectTime.Seconds())
	
	return keptIndices, discardedIndices, stats
}

func writePLY(filename string, splats []Splat, indices []int) error {
	file, err := os.Create(filename)
	if err != nil {
		return fmt.Errorf("creating output file: %v", err)
	}
	defer file.Close()

	// Write PLY header
	header := `ply
format binary_little_endian 1.0
element vertex %d
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float f_dc_0
property float f_dc_1
property float f_dc_2
property float f_rest_0
property float f_rest_1
property float f_rest_2
property float f_rest_3
property float f_rest_4
property float f_rest_5
property float f_rest_6
property float f_rest_7
property float f_rest_8
property float f_rest_9
property float f_rest_10
property float f_rest_11
property float f_rest_12
property float f_rest_13
property float f_rest_14
property float f_rest_15
property float f_rest_16
property float f_rest_17
property float f_rest_18
property float f_rest_19
property float f_rest_20
property float f_rest_21
property float f_rest_22
property float f_rest_23
property float f_rest_24
property float f_rest_25
property float f_rest_26
property float f_rest_27
property float f_rest_28
property float f_rest_29
property float f_rest_30
property float f_rest_31
property float f_rest_32
property float f_rest_33
property float f_rest_34
property float f_rest_35
property float f_rest_36
property float f_rest_37
property float f_rest_38
property float f_rest_39
property float f_rest_40
property float f_rest_41
property float f_rest_42
property float f_rest_43
property float f_rest_44
property float opacity
property float scale_0
property float scale_1
property float scale_2
property float rot_0
property float rot_1
property float rot_2
property float rot_3
end_header
`
	if _, err := fmt.Fprintf(file, header, len(indices)); err != nil {
		return err
	}

	// Write filtered splats
	for _, idx := range indices {
		if err := binary.Write(file, binary.LittleEndian, splats[idx]); err != nil {
			return err
		}
	}

	return nil
}

func computeSplatArea(s *Splat) float32 {
	// Using SuperSplat's formula: exp(s0)^2 + exp(s1)^2 + exp(s2)^2
	s0 := float32(math.Exp(float64(s.Scale0)))
	s1 := float32(math.Exp(float64(s.Scale1)))
	s2 := float32(math.Exp(float64(s.Scale2)))
	return s0*s0 + s1*s1 + s2*s2
}

func analyzeSplat(s *Splat) {
	fmt.Printf("Position: (%.6f, %.6f, %.6f)\n", s.X, s.Y, s.Z)
	fmt.Printf("Normal: (%.6f, %.6f, %.6f)\n", s.NX, s.NY, s.NZ)
	fmt.Printf("Scales: (%.6f, %.6f, %.6f)\n", s.Scale0, s.Scale1, s.Scale2)
	
	// Calculate exponentials
	s0 := float32(math.Exp(float64(s.Scale0)))
	s1 := float32(math.Exp(float64(s.Scale1)))
	s2 := float32(math.Exp(float64(s.Scale2)))
	fmt.Printf("Exponential Scales: (%.6f, %.6f, %.6f)\n", s0, s1, s2)
	
	// Calculate area using SuperSplat formula
	area := s0*s0 + s1*s1 + s2*s2
	fmt.Printf("SuperSplat Area: %.6f\n", area)
	
	fmt.Printf("\nRotation: (%.6f, %.6f, %.6f, %.6f)\n", s.Rot0, s.Rot1, s.Rot2, s.Rot3)
	fmt.Printf("Opacity: %.6f\n", s.Opacity)
	fmt.Printf("---\n")
}

func main() {
	startTime := time.Now()

	// Configuration
	config := FilterConfig{
		EnableNeighborFilter: true,
		Radius:              0.1,
		MinNeighbors:        10,
		EnableAreaFilter:    true,
		MaxArea:             0.05,  // Using SuperSplat's area calculation
		SaveDiscarded:       true,
		OutputDir:          "E:\\ECorals\\pipe\\q7\\2_train\\clean",
	}

	// Input/Output paths
	cellID := "0x-1y5s"
	colmapFile := fmt.Sprintf("E:\\ECorals\\pipe\\q7\\1_prep\\cells_s5m1\\%s\\sparse\\0\\points3D.bin", cellID)
	splatFile := fmt.Sprintf("E:\\ECorals\\pipe\\q7\\2_train\\crop\\crop-%s.ply", cellID)
	outputFile := fmt.Sprintf("%s\\c%s.ply", config.OutputDir, cellID)
	discardedFile := fmt.Sprintf("%s\\discarded-%s.ply", config.OutputDir, cellID)

	// Create output directory
	if err := os.MkdirAll(config.OutputDir, 0755); err != nil {
		fmt.Printf("Error creating output directory: %v\n", err)
		os.Exit(1)
	}

	// Read COLMAP points
	readStart := time.Now()
	fmt.Printf("Reading COLMAP points...\n")
	cloudPoints, err := readPoints3DBin(colmapFile)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Read %d points in %.1f seconds\n", len(cloudPoints), time.Since(readStart).Seconds())

	// Read splats
	splatsStart := time.Now()
	fmt.Printf("\nReading splats...\n")
	points, splats, err := readSplats(splatFile)
	if err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}
	fmt.Printf("Read %d splats in %.1f seconds\n", len(splats), time.Since(splatsStart).Seconds())

	// Filter cloud points by bounding box
	filterStart := time.Now()
	fmt.Printf("\nFiltering cloud points by bounds...\n")
	filteredPoints := filterPointsByBounds(cloudPoints, points, config.Radius)
	fmt.Printf("Filtered to %d points (%.1f%%) in %.1f seconds\n", 
		len(filteredPoints), 
		float64(len(filteredPoints))*100/float64(len(cloudPoints)),
		time.Since(filterStart).Seconds())

	// Build KD-tree
	treeStart := time.Now()
	fmt.Printf("\nBuilding KD-tree...\n")
	kdRoot := buildKDTree(filteredPoints, 0)
	fmt.Printf("Built KD-tree in %.1f seconds\n", time.Since(treeStart).Seconds())

	// Filter splats
	fmt.Printf("\nProcessing splats...\n")
	keptIndices, discardedIndices, stats := filterSplats(points, splats, kdRoot, config)

	// Write output files
	writeStart := time.Now()
	fmt.Printf("\nWriting filtered splats...\n")
	if err := writePLY(outputFile, splats, keptIndices); err != nil {
		fmt.Printf("Error: %v\n", err)
		os.Exit(1)
	}

	if config.SaveDiscarded && len(discardedIndices) > 0 {
		fmt.Printf("Writing discarded splats...\n")
		if err := writePLY(discardedFile, splats, discardedIndices); err != nil {
			fmt.Printf("Error writing discarded splats: %v\n", err)
		}
	}
	fmt.Printf("Wrote output files in %.1f seconds\n", time.Since(writeStart).Seconds())

	// Print statistics
	fmt.Printf("\nProcessing completed in %.1f seconds:\n", time.Since(startTime).Seconds())
	fmt.Printf("- Input points: %d\n", len(cloudPoints))
	fmt.Printf("- Filtered points: %d\n", len(filteredPoints))
	fmt.Printf("- Input splats: %d\n", stats.TotalSplats)
	fmt.Printf("- Output splats: %d (%.1f%%)\n", stats.KeptSplats, float64(stats.KeptSplats)*100/float64(stats.TotalSplats))
	
	if stats.TotalSplats-stats.KeptSplats > 0 {
		fmt.Printf("\nDiscarded splats by reason:\n")
		if config.EnableNeighborFilter {
			fmt.Printf("- By neighbors only: %d\n", stats.DiscardedByNeighbors)
		}
		if config.EnableAreaFilter {
			fmt.Printf("- By area only: %d\n", stats.DiscardedByArea)
		}
		if config.EnableNeighborFilter && config.EnableAreaFilter {
			fmt.Printf("- By both: %d\n", stats.DiscardedByBoth)
		}
	}
} 